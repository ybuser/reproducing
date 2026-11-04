"""Verify split backup pieces and restore the complete snapshot; Python standard library only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def verify():
    manifest = json.loads((ROOT/'backup_parts/manifest.json').read_text(encoding='utf-8'))
    whole = hashlib.sha256()
    size = 0
    for item in manifest['parts']:
        p = ROOT/'backup_parts'/item['name']
        if p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
            raise ValueError('Part checksum mismatch: '+item['name'])
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):
                whole.update(b)
                size += len(b)
    if whole.hexdigest() != manifest['zip_sha256'] or size != manifest['zip_bytes']:
        raise ValueError('Joined ZIP checksum mismatch')
    return manifest


def safe_target(base, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe ZIP member')
    p = base.joinpath(*rel.parts)
    if not p.resolve().is_relative_to(base.resolve()):
        raise ValueError('ZIP member escapes target')
    return p


def restore(destination):
    manifest = verify()
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Use an empty new destination; existing work will not be overwritten')
    destination.mkdir(parents=True,exist_ok=True)
    # Joined ZIP is temporary and is removed after verified extraction.
    fd, name = tempfile.mkstemp(prefix='thought-tracing-',suffix='.zip',dir=destination.parent)
    joined = Path(name)
    try:
        with os.fdopen(fd,'wb') as out:
            for item in manifest['parts']:
                with (ROOT/'backup_parts'/item['name']).open('rb') as f:
                    shutil.copyfileobj(f,out)
        if digest(joined) != manifest['zip_sha256']:
            raise ValueError('Joined ZIP checksum mismatch')
        with zipfile.ZipFile(joined) as z:
            seen = set()
            infos = z.infolist()
            for info in infos:
                p = safe_target(destination,info.filename)
                if info.filename in seen or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('Duplicate member or symbolic link')
                seen.add(info.filename)
                if p.exists():
                    raise FileExistsError(str(p))
            if z.testzip() is not None:
                raise ValueError('ZIP CRC validation failed')
            for info in infos:
                p = safe_target(destination,info.filename)
                if info.is_dir():
                    p.mkdir(parents=True,exist_ok=True)
                else:
                    p.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(info) as source,p.open('xb') as out:
                        shutil.copyfileobj(source,out)
        return {'restored_files':len(infos),'destination':str(destination),
                'next':'Enter thought_tracing and run: python tools/backup.py verify; python tools/backup.py restore-logs'}
    finally:
        joined.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','restore'])
    parser.add_argument('--destination',type=Path,default=ROOT/'.restored/full_snapshot')
    args=parser.parse_args()
    if args.command=='verify':
        m=verify()
        print(json.dumps({'passed':True,'parts':len(m['parts']),'zip_bytes':m['zip_bytes'],'zip_sha256':m['zip_sha256']},indent=2))
    else:
        print(json.dumps(restore(args.destination),indent=2))
