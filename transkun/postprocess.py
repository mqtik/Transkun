import pretty_midi


MIN_NOTE_DURATION = 0.03
DUPLICATE_MERGE_WINDOW = 0.05
MIN_VELOCITY = 5


def filter_notes(midi, min_duration=MIN_NOTE_DURATION, merge_window=DUPLICATE_MERGE_WINDOW, min_velocity=MIN_VELOCITY):
    for inst in midi.instruments:
        inst.notes = [n for n in inst.notes if n.velocity >= min_velocity]
        inst.notes = [n for n in inst.notes if (n.end - n.start) >= min_duration]

        inst.notes.sort(key=lambda n: (n.pitch, n.start))
        merged = []
        for note in inst.notes:
            if merged and merged[-1].pitch == note.pitch and note.start - merged[-1].end < merge_window:
                merged[-1].end = max(merged[-1].end, note.end)
                merged[-1].velocity = max(merged[-1].velocity, note.velocity)
            else:
                merged.append(note)
        inst.notes = merged

    return midi


def apply_sustain_pedal(midi, pedal_cc=64, pedal_threshold=64):
    for inst in midi.instruments:
        pedal_intervals = []
        pedal_start = None
        for cc in sorted(inst.control_changes, key=lambda c: c.time):
            if cc.number != pedal_cc:
                continue
            if cc.value >= pedal_threshold and pedal_start is None:
                pedal_start = cc.time
            elif cc.value < pedal_threshold and pedal_start is not None:
                pedal_intervals.append((pedal_start, cc.time))
                pedal_start = None

        if not pedal_intervals:
            continue

        for note in inst.notes:
            for p_start, p_end in pedal_intervals:
                if note.start >= p_start and note.start < p_end and note.end < p_end:
                    note.end = p_end
                    break

    return midi
