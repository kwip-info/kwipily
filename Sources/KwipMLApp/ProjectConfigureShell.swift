import SwiftUI

struct ProjectConfigureShell<Content: View>: View {
    var universal: Bool
    @ViewBuilder var content: (ProjectConfigurePage) -> Content
    @State private var selected: ProjectConfigurePage = .identity
    @State private var search = ""
    var body: some View {
        SettingsPageShell(title: "Configure", searchPlaceholder: "Search configuration",
            emptyTitle: "No matching configuration", clearSearchLabel: "Clear configuration search", identifierPrefix: "configure", footer: "This project",
            groups: ["Project", "Execution"], pages: ProjectConfigurePage.pages(universal: universal),
            matchingPages: ProjectConfigurePage.matches(search, universal: universal),
            subtitle: { $0.subtitle(universal: universal) }, selected: $selected, search: $search,
            content: content)
    }
}
