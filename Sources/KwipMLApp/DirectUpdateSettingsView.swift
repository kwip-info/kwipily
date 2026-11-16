import SwiftUI

struct DirectUpdateSettingsView:View {
    @ObservedObject private var updates=DirectUpdateStore.shared
    var body:some View {
        VStack(alignment:.leading,spacing:14) {
            SectionLabel(title:"Updates")
            Text(updates.message).font(.callout).textSelection(.enabled)
                .accessibilityIdentifier("settings.updates.status")
            Toggle("Check for updates automatically",isOn:Binding(get:{updates.automaticallyChecks},set:{updates.setAutomaticChecks($0)}))
                .disabled(!updates.available).accessibilityIdentifier("settings.updates.automatic")
            Button("Check for Updates…") {updates.check()}.disabled(!updates.canCheck)
                .accessibilityIdentifier("settings.updates.check")
            Text("Checks run in the background when due. Downloads require account access; your activated v1 license keeps working offline. Updates wait for active work to finish and use your normal project-save checks before restarting.")
                .font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
}
