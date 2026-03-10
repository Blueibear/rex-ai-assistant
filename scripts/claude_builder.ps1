param([string]$PacketFile)

$packet = Get-Content -Raw $PacketFile

$prompt = @"
You are the BUILDER agent.

Read the task packet.

Follow the ALLOWED_FILES list.

Do not inspect unrelated repo files.

Implement the change required to complete the task.

Run tests if needed.

Task packet:
$packet
"@

$prompt | claude