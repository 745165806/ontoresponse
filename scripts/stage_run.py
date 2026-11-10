"""Run a stage command with real exit code, time and persistent stdout/stderr."""
from pathlib import Path
import argparse,subprocess,time,json,datetime

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--cwd');p.add_argument('--out-dir',default='reports/next-stage-20261110');p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    command=args.command[1:] if args.command and args.command[0]=='--' else args.command
    root=Path(args.out_dir);root.mkdir(exist_ok=True)
    log=root/(args.name+'.log');start=time.time()
    with log.open('w') as output:r=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,cwd=args.cwd)
    result={'command':command,'exit_code':r.returncode,'started_utc':datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),
            'elapsed_seconds':time.time()-start,'log':str(log),'cwd':args.cwd or str(Path.cwd())}
    (root/(args.name+'.command.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return r.returncode
if __name__=='__main__':raise SystemExit(main())
