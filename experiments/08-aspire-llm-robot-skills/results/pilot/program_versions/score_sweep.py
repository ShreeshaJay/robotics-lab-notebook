import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + '/results.jsonl')]
for r in sorted(rows, key=lambda r: r['seed']):
    if not r['run_dir']:
        print(r['seed'], 'NO ARTIFACT', r['error']); continue
    o = json.load(open(r['run_dir'] + '/outcome.json'))
    print(r['seed'], o['program_ok'], o['task_success'], o['termination_reason'], o.get('steps_executed'), str(o.get('error'))[:160])
print('passed', sum(1 for r in rows if r['run_dir'] and
      json.load(open(r['run_dir'] + '/outcome.json'))['task_success']), 'of', len(rows))
