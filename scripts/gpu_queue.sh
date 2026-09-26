#!/usr/bin/env bash
# Wait until the GPU has >= MIN_FREE_MIB free (two readings 30 s apart),
# then run the given command. Progress lines go to STATUS for monitoring.
#   gpu_queue.sh NAME MIN_FREE_MIB STATUS LAUNCH_MARKER -- cmd...
NAME="$1"; MIN_FREE="$2"; STATUS="$3"; MARK="$4"; shift 5

free_mib() {
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '{print $1 - $2}'
}
recent_launch() {  # another queued job started < 90 s ago: its memory isn't visible yet
    [ -f "$MARK" ] && [ $(( $(date +%s) - $(stat -c %Y "$MARK") )) -lt 90 ]
}

echo "$(date +%T) $NAME QUEUED (needs ${MIN_FREE} MiB free, now $(free_mib))" >> "$STATUS"
while true; do
    if ! recent_launch && [ "$(free_mib)" -ge "$MIN_FREE" ]; then
        sleep 30
        if ! recent_launch && [ "$(free_mib)" -ge "$MIN_FREE" ]; then break; fi
    fi
    sleep 30
done
touch "$MARK"
echo "$(date +%T) $NAME STARTED ($(free_mib) MiB free)" >> "$STATUS"
"$@"
rc=$?
echo "$(date +%T) $NAME FINISHED exit=$rc" >> "$STATUS"
