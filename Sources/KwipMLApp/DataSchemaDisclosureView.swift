import SwiftUI

/// Shared visible provenance beside column editors; never writes a field draft.
struct DataSchemaDisclosureView: View {
    let disclosure: String
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Column metadata", systemImage: "tablecells").font(.system(size: 10, weight: .semibold))
            Text(disclosure).font(.system(size: 10)).lineSpacing(2).foregroundStyle(MLTheme.muted)
                .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
        }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
            .background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
    }
}
