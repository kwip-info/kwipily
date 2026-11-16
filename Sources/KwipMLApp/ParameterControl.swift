import SwiftUI

/// Commits only explicitly valid integers. Invalid text stays visible for correction.
struct ParameterControl: View {
    let title: String
    let hint: String
    let value: Double
    let minimum: Int
    let maximum: Int
    let enabled: Bool
    let commit: (Double) -> Void
    @State private var draft = ""
    @FocusState private var focused: Bool
    private var number: Int? { Int(draft) }
    private var valid: Bool { number.map { (minimum...maximum).contains($0) } ?? false }
    private var current: Int? { value.isFinite && value >= Double(minimum) && value <= Double(maximum) && value.rounded() == value ? Int(value) : nil }
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack { Text(title).font(.system(size: 12, weight: .medium)); Spacer(); Text("\(minimum)…\(maximum)").font(.system(size: 10, design: .monospaced)).foregroundStyle(MLTheme.muted) }
            HStack(spacing: 0) {
                Button { adjust(-1) } label: { Image(systemName: "minus").frame(width: 30,height: 30) }
                    .buttonStyle(.plain).disabled(!enabled || current == nil || current == minimum).accessibilityLabel("Decrease \(title)")
                Rectangle().fill(MLTheme.rule).frame(width: 1,height: 18)
                TextField(title,text:$draft).textFieldStyle(.plain).multilineTextAlignment(.center)
                    .font(.system(size: 13,design:.monospaced)).focused($focused).disabled(!enabled)
                    .accessibilityLabel(title).accessibilityHint("Whole number from \(minimum) to \(maximum). Press Return to apply.")
                    .onSubmit { apply() }
                Rectangle().fill(MLTheme.rule).frame(width:1,height:18)
                Button { adjust(1) } label: { Image(systemName:"plus").frame(width:30,height:30) }
                    .buttonStyle(.plain).disabled(!enabled || current == nil || current == maximum).accessibilityLabel("Increase \(title)")
            }.background(MLTheme.background).overlay(Rectangle().stroke(valid ? MLTheme.rule : Color.orange.opacity(0.8)))
            Text(!valid ? "Enter a whole number from \(minimum) to \(maximum)." : hint).font(.system(size:10)).foregroundStyle(valid ? MLTheme.muted : .orange).fixedSize(horizontal:false,vertical:true)
        }
        .onAppear { synchronize() }
        .onChange(of:value) { _,_ in if !focused { synchronize() } }
        .onChange(of:focused) { _,active in if !active { apply() } }
    }
    private func synchronize() { draft = current.map(String.init) ?? String(value) }
    private func apply() { guard enabled,valid,let number else { return }; commit(Double(number)) }
    private func adjust(_ delta:Int) { guard let current else { return }; let next=current+delta; guard (minimum...maximum).contains(next) else { return }; draft=String(next);commit(Double(next)) }
}
