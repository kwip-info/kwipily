import SwiftUI

struct BoundedNumberField: View {
    let title: String
    let value: NumericValue
    let domain: NumericDomain
    let origin: String
    let currentOrigin: () -> String?
    let applyValue: (NumericValue) -> Bool
    let onDraftChange: ((String, Bool) -> Void)?
    let onRawDraftChange: ((String) -> Void)?
    @State private var lifecycle = NumericDraftLifecycle()
    @State private var edit: NumericEditState
    @FocusState private var focused: Bool
    @Environment(\.isEnabled) private var enabled
    init(title: String, value: NumericValue, domain: NumericDomain, origin: String, currentOrigin: @escaping () -> String?, initialText: String? = nil, onRawDraftChange: ((String) -> Void)? = nil, onDraftChange: ((String, Bool) -> Void)? = nil, apply: @escaping (NumericValue) -> Bool) {
        self.title = title; self.value = value; self.domain = domain; self.origin = origin; self.currentOrigin = currentOrigin; self.applyValue = apply; self.onDraftChange = onDraftChange; self.onRawDraftChange = onRawDraftChange
        _edit = State(initialValue: NumericEditState(value: value, pendingText: initialText))
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                TextField(title, text: $edit.text).flatField().focused($focused).accessibilityLabel(title)
                    .accessibilityHint("Press Return or Apply to commit. Leaving this field also attempts to commit.").onSubmit(commit)
                Button("Apply", action: commit).buttonStyle(.borderless).accessibilityLabel("Apply \(title)")
            }
            if let error = edit.error { Text(error).font(.system(size: 10)).foregroundStyle(MLTheme.rose).fixedSize(horizontal: false, vertical: true).accessibilityLabel(error) }
        }
        .onChange(of: value) { _, next in if !focused && edit.error == nil { edit = NumericEditState(value: next) } }
        .onChange(of: origin) { old, _ in onDraftChange?(old, false); edit = NumericEditState(value: value); reportDraft() }
        .onAppear { lifecycle.appear(); onRawDraftChange?(edit.text); reportDraft() }
        .onDisappear { let report = lifecycle.disappear(origin: origin); onDraftChange?(report.origin, report.pending) }
        .onChange(of: edit.text) { _, text in if lifecycle.isMounted { onRawDraftChange?(text) }; reportDraft() }
        .onChange(of: edit.error) { _, _ in reportDraft() }
        .onChange(of: focused) { _, next in if !next { commit() } }
    }
    private func reportDraft() { if let report = lifecycle.report(origin: origin, pending: edit.needsCommit) { onDraftChange?(report.origin, report.pending) } }
    private func commit() {
        guard enabled, lifecycle.isMounted else { return }
        _ = edit.commit(domain: domain, origin: origin, currentOrigin: currentOrigin(), apply: applyValue)
        reportDraft()
    }
}
