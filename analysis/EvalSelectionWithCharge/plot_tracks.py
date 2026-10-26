#!/usr/bin/env python3
"""Gallery/PyROOT + Matplotlib event plots, following the ProtoDUNE workshop.
Source tools/gallery/setup.sh before running this script.
Reference: https://github.com/lhwhitehead/ProtoDUNEWorkshop/blob/main/dump_detsim_stage1.py
"""
from argparse import ArgumentParser
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-f', '--file', type=Path, default=repo / 'np02vd_raw_run043374_0000_df-s04-d0_dw_0_20260320T161028_cosmic_keepup.root')
    parser.add_argument('--entry', type=int, default=12, help='Zero-based event entry (default: 12)')
    parser.add_argument('--track-tag', default='pandoraTrack::pdvdkeepupstage1')
    parser.add_argument('--track-hit-tag', help='Track-hit association producer (default: --track-tag)')
    parser.add_argument('--shower-tag', default='pandoraShower::pdvdkeepupstage1')
    parser.add_argument('--raw-tag', default='tpcrawdecoder:daq:pdvdkeepupstage1')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output/python')
    parser.add_argument('--adc-limit', type=float, default=100, help='Raw color scale: +/- this ADC value')
    parser.add_argument('--tick-step', type=int, default=1, help='Display every Nth raw sample (default: every sample)')
    parser.add_argument('--skip-raw', action='store_true', help='Only plot reconstructed tracks/showers')
    parser.add_argument('--show', action='store_true', help='Also open Matplotlib windows')
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error(f'Input file does not exist: {args.file}')
    if args.entry < 0 or args.tick_step < 1 or args.adc_limit <= 0:
        parser.error('Entry must be nonnegative; tick-step and adc-limit must be positive')

    import ROOT as RT
    import numpy as np
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    RT.gROOT.SetBatch(True)
    for library in ('libgallery', 'liblardataobj_RecoBase', 'liblardataobj_RawData'):
        if RT.gSystem.Load(library) < 0:
            raise RuntimeError(f'Cannot load {library}; source tools/gallery/setup.sh first')
    # As in gallery_utils.py from the workshop: expose headers and explicitly
    # instantiate product access templates before calling them from Python.
    if not RT.gInterpreter.Declare('''
        #include "gallery/Event.h"
        #include "gallery/Handle.h"
        #include "gallery/ValidHandle.h"
        #include "canvas/Utilities/InputTag.h"
        #include "lardataobj/RecoBase/Track.h"
        #include "lardataobj/RecoBase/Shower.h"
        #include "lardataobj/RawData/RawDigit.h"
        #include "lardataobj/RawData/raw.h"
        #include "lardataobj/RecoBase/Hit.h"
        #include "canvas/Persistency/Common/FindManyP.h"
        #include <stdexcept>

        // Use the stored track-hit associations to compare in readout coordinates.
        std::vector<double> gallery_track_hit_coordinates(
            gallery::Event const& event,
            gallery::ValidHandle<std::vector<recob::Track>> const& tracks,
            art::InputTag const& tag) {
            art::FindManyP<recob::Hit> hits(tracks, event, tag);
            if (!hits.isValid()) throw std::runtime_error("Missing track-hit associations");
            std::vector<double> coordinates;
            for (std::size_t i = 0; i < tracks->size(); ++i)
                for (auto const& hit : hits.at(i)) {
                    coordinates.push_back(hit->Channel());
                    coordinates.push_back(hit->PeakTime());
                    coordinates.push_back(i);
                }
            return coordinates;
        }
    '''):
        raise RuntimeError('Cannot expose Gallery headers to PyROOT')
    for product in ('std::vector<recob::Track>', 'std::vector<recob::Shower>', 'std::vector<raw::RawDigit>'):
        if not RT.gInterpreter.Declare(
            f'template gallery::ValidHandle<{product}> gallery::Event::getValidHandle<{product}>(art::InputTag const&) const;'
        ):
            raise RuntimeError(f'Cannot instantiate Gallery access for {product}')
    if not RT.gInterpreter.Declare('''
        template bool gallery::Event::getByLabel<std::vector<recob::Shower>>(
            art::InputTag const&, gallery::Handle<std::vector<recob::Shower>>&) const;
    '''):
        raise RuntimeError('Cannot instantiate optional shower access')

    files = RT.std.vector('string')()
    files.push_back(str(args.file))
    event = RT.gallery.Event(files)
    if args.entry >= event.numberOfEventsInFile():
        parser.error(f'Entry {args.entry} is outside the file ({event.numberOfEventsInFile()} events)')
    event.goToEntry(args.entry)
    run, number = int(event.eventAuxiliary().run()), int(event.eventAuxiliary().event())
    tracks_handle = event.getValidHandle['std::vector<recob::Track>'](RT.art.InputTag(args.track_tag))
    tracks = tracks_handle.product()
    showers_handle = RT.gallery.Handle['std::vector<recob::Shower>']()
    event.getByLabel['std::vector<recob::Shower>'](RT.art.InputTag(args.shower_tag), showers_handle)
    showers = showers_handle.product() if showers_handle.isValid() else []
    if not showers_handle.isValid():
        print(f'No reconstructed shower collection: {args.shower_tag}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    track_label = 'Tracks'
    for track in tracks:
        points = [track.LocationAtPoint(i) for i in range(track.NumberTrajectoryPoints()) if track.HasValidPoint(i)]
        if points:
            ax.plot([p.Y() for p in points], [p.X() for p in points], color='tab:blue', linewidth=0.7, label=track_label)
            track_label = None
    shower_label = 'Showers (axis)'
    for shower in showers:
        start, direction = shower.ShowerStart(), shower.Direction()
        length = shower.Length() if shower.has_length() else 0.
        ax.plot([start.Y(), start.Y() + length * direction.Y()],
                [start.X(), start.X() + length * direction.X()],
                color='tab:red', marker='o', markersize=3, linewidth=1.5, label=shower_label)
        shower_label = None
    ax.set(xlabel='Y [cm]', ylabel='X [cm]', title=f'Run {run}, event {number}')
    ax.set_aspect('equal', adjustable='box')
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    path = args.output_dir / f'tracks_run{run}_event{number}_YX.png'
    fig.savefig(path, dpi=150)
    print(f'Plotted {len(tracks)} tracks and {len(showers)} showers: {path}')
    if not args.show:
        plt.close(fig)

    if not args.skip_raw:
        raw_handle = event.getValidHandle['std::vector<raw::RawDigit>'](RT.art.InputTag(args.raw_tag))
        digits = raw_handle.product()
        if len(digits):
            max_channel = max(int(d.Channel()) for d in digits)
            nticks = max(int(d.Samples()) for d in digits)
            nbins = (nticks + args.tick_step - 1) // args.tick_step
            coordinates = RT.gallery_track_hit_coordinates(
                event, tracks_handle, RT.art.InputTag(args.track_hit_tag or args.track_tag))
            track_hits = np.asarray(coordinates).reshape(-1, 3)
            extent = (-.5, max_channel + .5, -.5 * args.tick_step, (nbins - .5) * args.tick_step)
            fig, ax = plt.subplots(figsize=(16, 9))
            if len(track_hits):
                points = ax.scatter(track_hits[:, 0], track_hits[:, 1], c=track_hits[:, 2],
                                    cmap='turbo', s=3, linewidths=0, vmin=0, vmax=max(1, len(tracks) - 1))
                fig.colorbar(points, ax=ax, label='Track index')
            ax.set(xlim=extent[:2], ylim=extent[2:], xlabel='TPC channel', ylabel='Sample tick',
                   title=f'Run {run}, event {number}: reconstructed track-associated hits')
            fig.tight_layout()
            path = args.output_dir / f'tracks_run{run}_event{number}_channel_tick.png'
            fig.savefig(path, dpi=120)
            print(f'Plotted {len(track_hits)} track-associated hit peaks in raw readout coordinates: {path}')
            if not args.show:
                plt.close(fig)

            adc = np.full((max_channel + 1, nbins), np.nan, dtype=np.float32)
            samples = RT.std.vector('short')()
            for digit in digits:
                # ADCs() may contain compressed values: always decompress first.
                samples.resize(digit.Samples())
                RT.raw.Uncompress(digit.ADCs(), samples, digit.Compression())
                values = np.asarray(samples)[::args.tick_step]
                adc[int(digit.Channel()), :len(values)] = values.astype(np.float32) - digit.GetPedestal()
            fig, ax = plt.subplots(figsize=(16, 9))
            image = ax.imshow(adc.T, origin='lower', aspect='auto', interpolation='nearest',
                              extent=extent,
                              cmap='RdBu_r', vmin=-args.adc_limit, vmax=args.adc_limit)
            ax.set(xlabel='TPC channel', ylabel='Sample tick', title=f'Run {run}, event {number}: {args.raw_tag}')
            fig.colorbar(image, ax=ax, label='ADC - pedestal')
            fig.tight_layout()
            path = args.output_dir / f'rawTPC_run{run}_event{number}.png'
            fig.savefig(path, dpi=120)
            print(f'Plotted {len(digits)} decompressed raw TPC waveforms: {path}')
            if len(track_hits):
                ax.scatter(track_hits[:, 0], track_hits[:, 1], color='limegreen', s=3,
                           linewidths=0, label='Track-associated hit peaks')
                ax.legend(loc='upper right')
            ax.set(xlim=extent[:2], ylim=extent[2:],
                   title=f'Run {run}, event {number}: raw ADC + reconstructed track-associated hits')
            path = args.output_dir / f'rawTPC_tracks_run{run}_event{number}.png'
            fig.savefig(path, dpi=120)
            print(f'Saved raw/reconstructed-track comparison: {path}')
            if not args.show:
                plt.close(fig)
        else:
            print('No raw TPC digits in this event')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
