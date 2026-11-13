"""Queue authorised closure work to its existing owner; never open another writer.

Delivery and execution are distinct: a successful `codex queue` call only means
accepted. A per-ticket receipt records when the conversation starts and finishes.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time
import uuid


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def receipt(config_path, ticket, phase, now=None):
    if not re.fullmatch(r'[0-9a-f-]{36}', ticket):
        raise ValueError('Invalid ticket')
    base = Path(config_path).parent
    path = base / ('receipt-' + ticket + '.json')
    previous = json.loads(path.read_text()) if path.exists() else {}
    if phase == 'complete' and previous.get('status') != 'started':
        raise ValueError('Completion requires a start receipt')
    previous.update(ticket=ticket, status='started' if phase == 'ack' else 'complete',
                    updated_epoch=time.time() if now is None else now)
    save(path, previous)


def tick(config_path, runner=subprocess.run, now=None):
    supplied_time = now is not None
    now = time.time() if now is None else now
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    base = config_path.parent
    queue = Path(config['queue_state'])
    status_path = base / 'scheduler-status.json'
    with (base / 'scheduler.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 'FOLLOWUP_ALREADY_RUNNING'
        state = json.loads(queue.read_text())
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if not state.get('active', False):
            return 'QUEUE_INACTIVE'
        ticket = status.get('pending_ticket')
        if ticket:
            path = base / ('receipt-' + ticket + '.json')
            received = json.loads(path.read_text()) if path.exists() else {}
            if received.get('status') == 'complete':
                status.update(status='FOLLOWUP_COMPLETE', pending_ticket=None,
                              next_check_epoch=received['updated_epoch'] + config['interval_seconds'])
                save(status_path, status)
            else:
                started = received.get('status') == 'started'
                since = received.get('updated_epoch', status['queued_epoch'])
                ceiling = config.get('running_timeout_seconds', 7200) if started else config.get('delivery_timeout_seconds', 1800)
                label = 'FOLLOWUP_RUNNING' if started else 'FOLLOWUP_QUEUED'
                if now - since >= ceiling:
                    label = 'PAUSED_UNACKNOWLEDGED_EXECUTION' if started else 'PAUSED_UNACKNOWLEDGED_DELIVERY'
                    state.update(active=False, scheduler_blocker={'reason': label, 'ticket': ticket})
                    save(queue, state)
                status.update(status=label)
                save(status_path, status)
                return label
        if now < max(config['not_before_epoch'], status.get('next_check_epoch', 0)):
            return 'WAITING_FOR_SCHEDULE'
        attempt = int(status.get('attempts', 0)) + 1
        ticket = str(uuid.uuid4())
        log_path = base / ('delivery-%04d.log' % attempt)
        prompt = Path(config['prompt']).read_text()
        prefix = (f'[Authorised closure follow-up ticket {ticket}]\n'
                  f'Immediately acknowledge delivery with: python3 experiments/run_closure_followup.py '
                  f'--config {config_path} --ack {ticket}\n'
                  f'Before ending this turn, after recording progress, complete the receipt with: '
                  f'python3 experiments/run_closure_followup.py --config {config_path} --complete {ticket}\n'
                  'This message is for the existing conversation. Do not create another agent or scheduler.\n\n')
        # Queueing uses the existing owner and its permissions, not `exec resume`.
        command = [config['codex'], 'queue', '--thread', config['thread'],
                   '--message', prefix + prompt]
        with log_path.open('w') as log:
            try:
                result = runner(command, stdout=log, stderr=subprocess.STDOUT,
                                cwd=config['workspace'], timeout=30)
                rc = result.returncode
            except subprocess.TimeoutExpired:
                # Delivery may have succeeded; never enqueue an unacknowledged duplicate.
                log.write('Queue delivery timed out; acceptance is unknown.\n')
                rc = -2
            except Exception as error:
                log.write(type(error).__name__ + ': ' + str(error) + '\n')
                rc = -1
        ended = now if supplied_time else time.time()
        output = log_path.read_text(errors='replace')
        accepted = re.search(r'Queued message ([0-9a-f-]+) for thread ([0-9a-f-]+)', output)
        failures = 0 if rc == 0 and accepted else status.get('consecutive_failures', 0) + 1
        report = {'status': 'DELIVERY_FAILED', 'attempts': attempt, 'exit_code': rc,
                  'consecutive_failures': failures, 'log': str(log_path),
                  'finished_epoch': ended, 'next_check_epoch': ended + 120}
        if rc == 0 and accepted and accepted.group(2) == config['thread']:
            report.update(status='FOLLOWUP_QUEUED', pending_ticket=ticket,
                          message_id=accepted.group(1), queued_epoch=ended)
        elif rc in (0, -2) or failures >= 3:
            report['status'] = 'PAUSED_DELIVERY_UNCERTAIN' if rc in (0, -2) else 'PAUSED_AFTER_THREE_FAILURES'
            state.update(active=False, scheduler_blocker=report)
            save(queue, state)
        save(status_path, report)
        return report['status']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--ack')
    action.add_argument('--complete')
    args = parser.parse_args()
    if args.ack or args.complete:
        receipt(args.config, args.ack or args.complete, 'ack' if args.ack else 'complete')
        print('RECEIPT_RECORDED', flush=True)
    else:
        print(tick(args.config), flush=True)
