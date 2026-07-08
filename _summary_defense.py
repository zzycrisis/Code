import json
import os
import glob

results = []
for path in sorted(glob.glob('results/defense/**/*.json', recursive=True)):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    fname = os.path.basename(path).replace('.json', '')
    defense_dir = os.path.basename(os.path.dirname(path))

    jb_name = data.get('jailbreak method', '?')
    jb_sr = data.get('jailbreak success rate', 0) * 100
    pass_dr = data.get('pass defense rate', 0) * 100
    asr = data.get('attack success rate', 0) * 100
    resp_delay = data.get('average response delay', 0)
    def_delay = data.get('average defense delay', 0)
    stack_delay = data.get('average stack delay', 0)

    results.append({
        'file': fname,
        'defense': defense_dir,
        'method': jb_name,
        'jb_sr': jb_sr,
        'pass_dr': pass_dr,
        'asr': asr,
        'resp_delay': resp_delay,
        'def_delay': def_delay,
        'stack_delay': stack_delay,
    })

# 按 defense 类型分组打印
for defense_type in sorted(set(r['defense'] for r in results)):
    subset = [r for r in results if r['defense'] == defense_type]
    print(f"\n{'='*90}")
    print(f"  Defense: {defense_type}")
    print(f"{'='*90}")
    print(f"{'Method':<18s} {'ASR(before)':>10s} {'ASR(after)':>10s} {'Blocked':>8s} {'Stack Delay':>10s}")
    print(f"{'-'*56}")

    total_jb, total_asr, total_stack = 0, 0, 0
    for r in sorted(subset, key=lambda x: -x['jb_sr']):
        blocked = r['jb_sr'] - r['asr']
        print(f"{r['method']:<18s} {r['jb_sr']:>9.3f}% {r['asr']:>9.3f}% {blocked:>7.3f}% {r['stack_delay']:>9.2f}s")
        total_jb += r['jb_sr']
        total_asr += r['asr']
        total_stack += r['stack_delay']

    n = len(subset)
    avg_blocked = (total_jb - total_asr) / n
    print(f"{'-'*56}")
    print(f"{'AVERAGE':<18s} {total_jb/n:>9.3f}% {total_asr/n:>9.3f}% {avg_blocked:>7.3f}% {total_stack/n:>9.2f}s")

# 缺失文件
print(f"\n{'='*60}")
print("Files NOT yet evaluated:")
print(f"{'='*60}")

expected_methods = [
    'GCG', 'AutoDAN-GA', 'PAIR', 'DrAttack', 'LLM-Fuzzer',
    'TAP', 'RLbreaker', 'Puzzler', 'DAN', 'MultiJail'
]
for defense_type in sorted(set(r['defense'] for r in results)):
    done = set(r['method'] for r in results if r['defense'] == defense_type)
    missing = [m for m in expected_methods if m not in done]
    if missing:
        print(f"  {defense_type}:")
        for m in missing:
            print(f"    - {m}")
    else:
        print(f"  {defense_type}: COMPLETE")
