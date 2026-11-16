#if KWIPML_RECOVERY_UAT
import SwiftUI

struct RecoveryUATCommands: Commands {
    @ObservedObject private var gate = RecoveryUATGate.shared
    var body: some Commands {
        CommandMenu("Recovery UAT") {
            Text("Diagnostic build · 5-minute pause limit")
            ForEach(RecoveryUATGate.Stage.allCases, id: \.self) { stage in
                Button("Arm: \(stage.rawValue)") { gate.arm(stage) }
                    .disabled(gate.reached != nil)
            }
            if let armed = gate.armed {
                Text("Armed: \(armed.rawValue)")
                Button("Disarm") { gate.disarm() }
            }
            if let reached = gate.reached {
                Divider()
                Text("Reached: \(reached.stage.rawValue)")
                Text("Group: \(reached.identity.groupID)")
                Text("Run: \(reached.identity.runID)")
                Text("Variant: \(reached.identity.ordinal)")
                if let nonce = reached.identity.nonce { Text("Launch: \(nonce)") }
                Button("Resume paused work") { gate.resume() }
                Button("Cancel paused work") { gate.cancel(reached.token) }
            }
        }
    }
}
#endif
