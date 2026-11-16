import SwiftUI
import AppKit
import UniformTypeIdentifiers
import KwipMLCore

struct SupportDiagnosticSettingsView: View {
    @ObservedObject private var updates = DirectUpdateStore.shared
    private struct PreparedReport: Identifiable { let id = UUID(); let data: Data }
    @State private var prepared: PreparedReport?
    @State private var message: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionLabel(title: "Support")
            Text("Get help with \(ProductBrand.name)").font(.headline)
            Link("Contact contact@kwip.info", destination: URL(string: "mailto:contact@kwip.info")!)
            Text("Support is provided as time permits. Include your app version and a short description of the issue.").font(.caption).foregroundStyle(MLTheme.muted)
            Link("Support guide", destination: URL(string: "https://kwipily.kwip.info/support/")!)
            Link("Privacy", destination: URL(string: "https://kwipily.kwip.info/privacy/")!)
            Divider()
            SectionLabel(title: "Support report")
            Text("Review a small snapshot of your app version, macOS version and the latest update outcome in this session.")
                .font(.callout)
            Text("No projects, model data, logs, file paths, account details, license files or credentials are included. Nothing is sent automatically. You choose whether to save and share the report.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            Button("Review Support Report…") { review() }
                .accessibilityIdentifier("settings.support.review")
            if let message { Text(message).font(.caption).textSelection(.enabled).accessibilityIdentifier("settings.support.status") }
        }
        .sheet(item: $prepared) { report in
            VStack(alignment: .leading, spacing: 14) {
                Text("Review Support Report").font(.title2)
                Text("This is the complete report. Update outcomes describe this session. Saving makes a local file only.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                ScrollView {
                    Text(String(decoding: report.data, as: UTF8.self))
                        .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading).padding(12)
                }.frame(height: 300).overlay(Rectangle().stroke(MLTheme.rule))
                    .accessibilityIdentifier("settings.support.preview")
                if let message { Text(message).font(.caption).foregroundStyle(MLTheme.rose).accessibilityIdentifier("settings.support.saveStatus") }
                HStack {
                    Button("Cancel", role: .cancel) { prepared = nil }.keyboardShortcut(.cancelAction)
                    Spacer()
                    Button("Save Report…") { save(report.data) }.keyboardShortcut(.defaultAction)
                        .accessibilityIdentifier("settings.support.save")
                }
            }.padding(24).frame(width: 520).buttonStyle(FlatButtonStyle())
        }
    }
    private func review() {
        #if arch(arm64)
        let architecture = SupportDiagnosticReport.Architecture.arm64
        #elseif arch(x86_64)
        let architecture = SupportDiagnosticReport.Architecture.x86_64
        #else
        let architecture = SupportDiagnosticReport.Architecture.unknown
        #endif
        let report = SupportDiagnosticReport(
            version: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String,
            build: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String,
            os: ProcessInfo.processInfo.operatingSystemVersion, architecture: architecture,
            edition: DesktopDeployment.isCommercialEdition(bundle: Bundle.main.bundleIdentifier) && !DesktopDeployment.isStaging ? .direct : .development,
            updaterAvailable: updates.available, automaticChecks: updates.automaticallyChecks,
            updateOutcome: updates.diagnosticOutcome)
        do { prepared = PreparedReport(data: try report.encoded()); message = nil }
        catch { message = "Could not prepare a support report. Try again or contact support." }
    }
    private func save(_ data: Data) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "Kwip-ML-Support.json"; panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            try data.write(to: url, options: .atomic)
            prepared = nil; message = "Support report saved. Nothing was sent. Share it with support only if you choose to."
        } catch { message = "Could not save the report. Choose a writable folder and retry." }
    }
}
