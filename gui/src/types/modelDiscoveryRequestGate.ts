export class ModelDiscoveryRequestGate {
  private latestRequestId = 0

  begin(): number {
    this.latestRequestId += 1
    return this.latestRequestId
  }

  invalidate(): void {
    this.latestRequestId += 1
  }

  isCurrent(requestId: number): boolean {
    return requestId === this.latestRequestId
  }
}
