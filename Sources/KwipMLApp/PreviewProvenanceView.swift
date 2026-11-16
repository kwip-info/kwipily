import SwiftUI

struct PreviewProvenanceView: View {
    let values: [String: Any]
    var body: some View {
        if values["inferenceOnly"] as? Bool == true {
            VStack(alignment: .leading, spacing: 7) {
                Text("Preview inputs · \(values["count"] as? Int ?? 0) validation rows · inference only")
                    .font(.system(size: 11, weight: .medium))
                Text("A bounded stage preview, not an accuracy or generalization estimate.")
                    .font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                DisclosureGroup("Input rows & provenance") {
                    VStack(alignment: .leading, spacing: 6) {
                        if let names = values["featureColumns"] as? [String] {
                            Text("Features: " + names.joined(separator: ", "))
                        }
                        if let rows = values["rowIDs"] as? [Any] {
                            Text("Observation IDs: " + rows.prefix(8).map { String(describing: $0) }.joined(separator: ", "))
                        }
                        if let preparation = values["preparation"] as? [String: Any],
                           let digest = preparation["manifestSHA256"] as? String {
                            Text("Frozen preparation SHA-256: " + digest)
                        }
                    }.font(.system(size: 10, design: .monospaced)).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 5)
                }.font(.system(size: 11)).tint(MLTheme.teal)
            }
        }
    }
}
