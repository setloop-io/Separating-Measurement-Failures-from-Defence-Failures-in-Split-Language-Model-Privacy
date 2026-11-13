"""Build provisional arXiv source packages without submitting or changing papers."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/arxiv-preprints-20260920',
                        help='Fresh output directory; existing packages are never overwritten')
    args = parser.parse_args()
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in ('paper_a', 'paper_b'):
        original = ROOT / 'paper' / name
        target = base / name
        source = target / 'source'
        source.mkdir(parents=True, exist_ok=False)
        tex = (original / 'main.tex').read_text()
        figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)
        for figure in figures:
            p = (original / figure).resolve()
            destination = source / 'figures' / p.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(p, destination)
            tex = tex.replace('{' + figure + '}', '{figures/' + p.name + '}')
        (source / 'main.tex').write_text(tex)
        for filename in ('main.bbl', 'main.bib'):
            shutil.copy2(original / filename, source / filename)
        inputs = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
        # Build outside the source payload so log/auxiliary/PDF files never enter its zip.
        check = target / 'compile-check'
        shutil.copytree(source, check)
        for i in (1, 2):
            with (target / ('compile-%d.log' % i)).open('wb') as log:
                subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                               cwd=check, stdout=log, stderr=subprocess.STDOUT, check=True)
        log = (target / 'compile-2.log').read_text()
        if any(term in log for term in ('undefined references', 'undefined citations', 'Overfull \\hbox', 'LaTeX Warning')):
            raise ValueError('LaTeX review required: ' + name)
        shutil.copy2(check / 'main.pdf', target / (name + '.pdf'))
        archive = target / (name + '-arxiv-source.zip')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as package:
            for relative in sorted(inputs):
                package.write(source / relative, relative)
        abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S).group(1).strip()
        title = re.search(r'\\title\{([^}]+)\}', tex).group(1)
        metadata = {'title': title, 'abstract_latex': abstract,
                    'authors_and_affiliations': 'Preserved in main.tex; author confirmation required before submission',
                    'primary_category_proposal': 'cs.CR', 'cross_list_proposal': 'cs.LG if appropriate after author review',
                    'licence': 'AUTHOR_SELECTION_REQUIRED',
                    'status': 'PROVISIONAL_PENDING_CLOSURE_QUEUE_AND_FINAL_AUTHOR_METADATA',
                    'submission_performed': False,
                    'source_sha256': inputs, 'archive_sha256': sha(archive),
                    'pdf_sha256': sha(target / (name + '.pdf')),
                    'build': 'Two successful PDFLaTeX passes; no undefined references/citations or overfull boxes',
                    'remaining': ['Update final evidence links and any reviewed follow-up findings',
                                  'Confirm authors/affiliations, category and licence',
                                  'Review final rendered PDF and arXiv-generated preview before actual submission'],
                    'official_guidance': 'https://info.arxiv.org/help/submit_tex.html'}
        (target / 'METADATA.json').write_text(json.dumps(metadata, indent=2) + '\n')
        outputs[name] = {'source_archive': str(archive),
                         'pdf': str(target / (name + '.pdf')),
                         'source_files': len(inputs), 'status': metadata['status']}
    (base / 'INDEX.json').write_text(json.dumps(outputs, indent=2) + '\n')
    print(json.dumps(outputs, indent=2))

if __name__ == '__main__':
    main()
