#!/usr/bin/env python3
"""Plot reconstructed space points with Y horizontal and X vertical.
Source tools/gallery/setup.sh before running this script.
"""
from argparse import ArgumentParser
from pathlib import Path

def main():
    repo = Path(__file__).resolve().parents[2]
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-f', '--file', type=Path, default=repo / 'np02vd_raw_run043374_0000_df-s04-d0_dw_0_20260320T161028_cosmic_keepup.root')
    parser.add_argument('--entry', type=int, default=12, help='Zero-based event entry (default: 12)')
    parser.add_argument('--spacepoint-tag', '--tag', default='pandora::pdvdkeepupstage1', help='Space-point product tag')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output/python')
    parser.add_argument('--show', action='store_true', help='Also open the Matplotlib window')
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error(f'Input file does not exist: {args.file}')
    if args.entry < 0:
        parser.error('Entry must be nonnegative')

    import ROOT as RT
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dunestyle.matplotlib as dunestyle

    RT.gROOT.SetBatch(True)
    for library in ('libgallery', 'liblardataobj_RecoBase'):
        if RT.gSystem.Load(library) < 0:
            raise RuntimeError(f'Cannot load {library}; source tools/gallery/setup.sh first')
    product = 'std::vector<recob::SpacePoint>'
    if not RT.gInterpreter.Declare('''
        #include "gallery/Event.h"
        #include "gallery/ValidHandle.h"
        #include "canvas/Utilities/InputTag.h"
        #include "lardataobj/RecoBase/SpacePoint.h"
    ''' + f'template gallery::ValidHandle<{product}> gallery::Event::getValidHandle<{product}>(art::InputTag const&) const;'):
        raise RuntimeError('Cannot expose Gallery space-point access to PyROOT')

    files = RT.std.vector('string')()
    files.push_back(str(args.file))
    event = RT.gallery.Event(files)
    if args.entry >= event.numberOfEventsInFile():
        parser.error(f'Entry {args.entry} is outside the file ({event.numberOfEventsInFile()} events)')
    event.goToEntry(args.entry)
    handle = event.getValidHandle[product](RT.art.InputTag(args.spacepoint_tag))
    points = handle.product()
    run, number = int(event.eventAuxiliary().run()), int(event.eventAuxiliary().event())

    y, x = [], []
    for point in points:
        position = point.position()
        y.append(float(position.Y()))
        x.append(float(position.X()))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(y, x, s=1, color='tab:blue', linewidths=0)
    ax.set(xlabel='Y [cm]', ylabel='X [cm]',
           title=f'Run {run}, event {number}: space points ({args.spacepoint_tag})')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    producer = args.spacepoint_tag.split(':')[0]
    path = args.output_dir / f'spacepoints_{producer}_run{run}_event{number}_YX.png'
    fig.savefig(path, dpi=150)
    print(f'Plotted {len(points)} space points: {path}')
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
