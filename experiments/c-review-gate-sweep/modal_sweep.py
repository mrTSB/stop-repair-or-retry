"""Modal CPU job: fetch Tanvir's abc-runtime archive, replay latest, sweep C's review gate.

Run from this directory: modal run --detach modal_sweep.py --run-name run1
Outputs land in Modal volume `cs238-c-sweep` under /out/<run>/ (log.txt, sweep.json).
If the sweep step fails, the container stays up for 45 min and runs /out/fix/fix.sh
(cwd = extracted bundle, $OUT = output dir) whenever that file is re-uploaded.
"""
import modal

RELEASE = 'https://github.com/mrTSB/stop-repair-or-retry/releases/download/abc-runtime-2026-11-25'
HERE = __import__('pathlib').Path(__file__).parent

image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('zstd', 'curl', 'ca-certificates')
         .pip_install('uv', 'requests')
         .add_local_file(HERE.parents[1] / 'scripts/reproduction/requirements.txt', '/opt/requirements.txt', copy=True)
         .run_commands('uv venv /opt/venv --python 3.12.11',
                       'uv pip install --python /opt/venv/bin/python -r /opt/requirements.txt')
         .add_local_file(HERE / 'sweep_c.py', '/opt/sweep_c.py'))
volume = modal.Volume.from_name('cs238-c-sweep', create_if_missing=True)
app = modal.App('cs238-c-sweep', image=image)


@app.function(cpu=8, memory=16384, ephemeral_disk=524288, timeout=3 * 3600, volumes={'/out': volume})
def sweep(run_name: str):
    import hashlib, json, os, shutil, subprocess, time
    from concurrent.futures import ThreadPoolExecutor
    import requests

    out = f'/out/{run_name}'
    os.makedirs(out, exist_ok=True)

    def say(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(f'{out}/log.txt', 'a') as log: log.write(line + '\n')
        volume.commit()

    work = '/work'; parts_dir = f'{work}/parts'; os.makedirs(parts_dir, exist_ok=True)
    meta = requests.get(f'{RELEASE}/archive.json', timeout=60).json()
    parts = meta['parts']
    say(f'{len(parts)} parts, {meta["archive_size_bytes"]/1e9:.1f} GB')

    def fetch(p):
        dest = f'{parts_dir}/{p["name"]}'
        for attempt in range(4):
            h = hashlib.sha256()
            with requests.get(f'{RELEASE}/{p["name"]}', stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk); h.update(chunk)
            if h.hexdigest() == p['sha256'] and os.path.getsize(dest) == p['size_bytes']:
                return p['name']
            say(f'checksum mismatch {p["name"]} attempt {attempt}')
        raise RuntimeError(f'failed {p["name"]}')

    t0 = time.time()
    with ThreadPoolExecutor(6) as pool:
        for i, name in enumerate(pool.map(fetch, parts), 1):
            if i % 6 == 0: say(f'downloaded+verified {i}/{len(parts)} ({time.time()-t0:.0f}s)')
    say(f'all parts verified ({time.time()-t0:.0f}s)')

    os.makedirs(f'{work}/x', exist_ok=True)
    names = ' '.join(f'{parts_dir}/{p["name"]}' for p in parts)
    subprocess.run(f'cat {names} | zstd -d -T0 | tar -xf - -C {work}/x', shell=True, check=True)
    shutil.rmtree(parts_dir)
    say(f'extracted ({time.time()-t0:.0f}s)')

    root = f'{work}/x/stop-repair-or-retry'
    shutil.copy('/opt/sweep_c.py', f'{root}/scripts/reproduction/sweep_c.py')
    uvrun = '/opt/venv/bin/python scripts/reproduction/run.py'
    env = {k: v for k, v in os.environ.items() if k != 'VIRTUAL_ENV'}

    def step(name, cmd):
        say(f'start {name}')
        p = subprocess.run(cmd, shell=True, cwd=root, env=env, capture_output=True, text=True)
        with open(f'{out}/{name}.stdout.txt', 'w') as fh: fh.write(p.stdout[-200000:] + '\n--- stderr ---\n' + p.stderr[-200000:])
        say(f'{name} exit {p.returncode}: {p.stdout[-600:]} {p.stderr[-1500:] if p.returncode else ""}')
        return p.returncode

    step('replay', f'{uvrun} scripts/reproduction/replay_latest.py --output runs/replay-latest')
    for f in ('summary.json',):
        src = f'{root}/runs/replay-latest/{f}'
        if os.path.exists(src): shutil.copy(src, f'{out}/replay-{f}')
    rc = step('sweep', f'{uvrun} scripts/reproduction/sweep_c.py --output runs/c-sweep0 --workers 8')
    if os.path.exists(f'{root}/runs/c-sweep0/sweep.json'):
        shutil.copy(f'{root}/runs/c-sweep0/sweep.json', f'{out}/sweep.json')
    # Recovery hook: run /out/fix/fix.sh (cwd=root, $OUT=out dir) each time it changes; idle exit after 45 min.
    fix = '/out/fix/fix.sh'; seen = None; deadline = time.time() + 45 * 60; attempt = 0
    while rc != 0 and time.time() < deadline:
        time.sleep(30); volume.reload()
        if os.path.exists(fix) and os.path.getmtime(fix) != seen:
            seen = os.path.getmtime(fix); attempt += 1
            if os.path.exists('/out/fix/sweep_c.py'):
                shutil.copy('/out/fix/sweep_c.py', f'{root}/scripts/reproduction/sweep_c.py')
            env['OUT'] = out; env['UVRUN'] = uvrun
            rc = step(f'fix{attempt}', f'bash {fix}')
            deadline = time.time() + 45 * 60
    say(f'finished rc={rc} total {time.time()-t0:.0f}s')
    volume.commit()


@app.local_entrypoint()
def main(run_name: str = 'run1'):
    sweep.remote(run_name)
