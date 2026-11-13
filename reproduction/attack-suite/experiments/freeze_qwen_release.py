"""Archive a completed Qwen run; publication remains gated on CPU reproduction."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


class Parts:
    def __init__(self,root,limit=4*1024**3):
        self.root=root;self.limit=limit;self.stream=None;self.index=0;self.size=0;self.records=[]
    def write(self,data):
        total=len(data)
        while data:
            if self.stream is None:
                self.path=self.root/('evidence.tar.gz.part%03d'%self.index)
                self.stream=self.path.open('xb');self.index+=1;self.size=0;self.hash=hashlib.sha256()
            n=min(len(data),self.limit-self.size);block=data[:n]
            self.stream.write(block);self.hash.update(block);self.size+=n;data=data[n:]
            if self.size==self.limit:self.close_part()
        return total
    def close_part(self):
        if self.stream is not None:
            self.stream.close();self.records.append({'path':self.path.name,'bytes':self.size,'sha256':self.hash.hexdigest()});self.stream=None
    def flush(self):
        if self.stream is not None:self.stream.flush()
    def close(self):self.close_part()


class RecordingReader:
    def __init__(self,source):self.source=source;self.hash=hashlib.sha256();self.size=0
    def read(self,n=-1):
        block=self.source.read(n);self.hash.update(block);self.size+=len(block);return block


def freeze(source,output,limit=4*1024**3):
    status=json.loads((source/'status.json').read_text())
    marker=json.loads((source/'grid/grid-complete.json').read_text())
    analysis=json.loads((source/'grid.analysis.json').read_text())
    pin='981889c4df947496fbfaa6e138510ee031dc4bf65ab26e5169763c6131595085'
    if status['status']!='COMPLETE_REVIEW_AND_PUBLICATION_REQUIRED' or marker.get('count')!=200 or marker.get('canary') or marker['protocol_sha256']!=pin or analysis['protocol_sha256']!=pin:
        raise ValueError('Complete frozen grid required')
    anchors={Path(k).parent.name:v for k,v in analysis['evidence_anchors'].items()}
    cells=list((source/'grid').glob('*/cell.json'))
    if len(anchors)!=200 or len(cells)!=200 or set(anchors)!={p.parent.name for p in cells}:raise ValueError('Incomplete cell anchors')
    for p in cells:
        if sha(p)!=anchors[p.parent.name]:raise ValueError('Cell anchor changed')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();writer=Parts(output,limit);count=0;size=0;grid_count=0
    def record(state,**extra):
        value={'status':state,'files':count,'raw_bytes':size,'wall_seconds':time.monotonic()-started,**extra}
        save(output/'ARCHIVE-STATUS.json',value);print(json.dumps(value),flush=True)
    record('ARCHIVING_COMPLETED_EVIDENCE')
    try:
        with (output/'RAW_FILES.jsonl').open('x') as inventory,gzip.GzipFile(fileobj=writer,mode='wb',compresslevel=1,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w|') as archive:
            for path in sorted(source.rglob('*')):
                if path.is_symlink():raise ValueError('Symlink in evidence')
                if not path.is_file():continue
                rel=path.relative_to(source).as_posix();before=path.stat()
                info=archive.gettarinfo(str(path),arcname='qwen-fp64-v2/'+rel)
                with path.open('rb') as f:
                    reader=RecordingReader(f);archive.addfile(info,reader)
                after=path.stat()
                if reader.size!=before.st_size or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Evidence changed during archive')
                row={'path':rel,'bytes':reader.size,'sha256':reader.hash.hexdigest()};inventory.write(json.dumps(row)+'\n')
                count+=1;size+=reader.size;grid_count+=int(rel.startswith('grid/'))
                if rel.endswith('/cell.json') and rel.startswith('grid/') and row['sha256']!=anchors[path.parent.name]:raise ValueError('Archived anchor mismatch')
                if count%1000==0:record('ARCHIVING_COMPLETED_EVIDENCE')
        writer.close()
        report={'status':'ARCHIVED_PENDING_CPU_REPRODUCTION','source':str(source),'source_run':'766a85bc-0bfe-40d2-aa13-51df27f5e926','source_commit':'e2d4268e900b7445d943906b8a25221457aeaaaf','protocol_sha256':pin,'verified_cell_anchors':200,'raw_files':count,'grid_files':grid_count,'raw_bytes':size,'parts':writer.records,'raw_inventory_sha256':sha(output/'RAW_FILES.jsonl'),'analysis_sha256':sha(source/'grid.analysis.json'),'wall_seconds':time.monotonic()-started,'canary_scope':'The two excluded canary directories are retained as history, never members of the 200-cell comparison.','publication_gate':'Require successful independent CPU reproduction and exact grid inventory hash agreement before HF upload or paper claims.'}
        save(output/'ARCHIVE.json',report);record('ARCHIVED_PENDING_CPU_REPRODUCTION',parts=len(writer.records),compressed_bytes=sum(x['bytes'] for x in writer.records))
    except Exception as error:
        writer.close();record('FAILED',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze(a.source,a.output)
