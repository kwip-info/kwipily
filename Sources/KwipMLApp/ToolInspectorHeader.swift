import SwiftUI
import KwipMLCore

private struct InspectedComponentTypeKey: EnvironmentKey { static let defaultValue: String? = nil }
private struct InspectedComponentConfigurationKey: EnvironmentKey { static let defaultValue: [String: JSONValue] = [:] }
extension EnvironmentValues {
    var inspectedComponentType: String? {
        get { self[InspectedComponentTypeKey.self] }
        set { self[InspectedComponentTypeKey.self] = newValue }
    }
    var inspectedComponentConfiguration: [String: JSONValue] {
        get { self[InspectedComponentConfigurationKey.self] }
        set { self[InspectedComponentConfigurationKey.self] = newValue }
    }
}

struct ToolInspectorHeader: View {
    let title: String
    let authoredName: String
    let iconType: String
    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            ComponentIcon(type: iconType, size: 32)
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.system(size: 16, weight: .semibold))
                    .accessibilityAddTraits(.isHeader).accessibilityIdentifier("inspector.tool.title")
                if !authoredName.isEmpty && authoredName != title {
                    Text(authoredName).font(.caption).foregroundStyle(MLTheme.muted)
                }
            }.fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}
