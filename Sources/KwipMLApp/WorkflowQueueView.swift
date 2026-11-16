import SwiftUI

struct WorkflowQueueView: View {
    @ObservedObject var queue: WorkflowExecutionQueue
    @State private var expanded = true
    var body: some View {
        DisclosureGroup("Training queue · \(queue.entries.filter { !$0.state.terminal && $0.state != .paused }.count) active or waiting", isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 8) {
                Text("One slot shared by PyTorch Train requests and experiment groups across project windows. Backend fit and frozen evaluation use their own existing execution controls. Waiting does not consume the worker time allowance.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                if let error = queue.error { Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled) }
                if queue.entries.isEmpty { Text("No captured training requests.").font(.caption).foregroundStyle(MLTheme.muted) }
                else {
                    ThemedScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(queue.entries.reversed()) { entry in
                                HStack(alignment: .top) {
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text("\(entry.capture.title) · \(entry.state.rawValue.capitalized)").font(.system(size: 12, weight: .medium))
                                        Text("\(entry.capture.kind.rawValue.capitalized) · \(entry.capture.requestIdentities.count) request(s) · source revision \(entry.capture.documentRevision)")
                                            .font(.caption).foregroundStyle(MLTheme.muted)
                                        if let location = entry.capture.externalLocationID {
                                            Text("Linked location · " + location.uuidString.prefix(12)).font(.caption).foregroundStyle(MLTheme.muted)
                                        }
                                        Text(entry.message).font(.caption).foregroundStyle(entry.state == .failed ? MLTheme.rose : MLTheme.muted).textSelection(.enabled)
                                        if entry.state == .paused {
                                            Text("Open the original project, review its current source/settings, then submit again. That creates a new capture; this receipt cannot resume old inputs.").font(.caption).foregroundStyle(MLTheme.muted)
                                        }
                                    }
                                    Spacer()
                                    if !entry.state.terminal {
                                        Button(entry.state == .paused ? "Dismiss" : "Cancel") { queue.cancel(entry.id) }
                                            .buttonStyle(FlatButtonStyle()).accessibilityLabel("Cancel queued request " + entry.capture.title)
                                            .disabled(entry.state == .cancelling)
                                    }
                                }
                            }
                        }
                    }.frame(maxHeight: 180)
                }
                Button("Clear finished queue entries") { queue.clearFinished() }
                    .buttonStyle(FlatButtonStyle()).disabled(!queue.entries.contains(where: {$0.state.terminal}))
                Text("Queue receipts are separate from saved run History. Clearing finished entries does not remove run artifacts.").font(.caption).foregroundStyle(MLTheme.muted)
            }.padding(.top, 8)
        }
    }
}
