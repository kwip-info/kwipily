import SwiftUI

/// Shared visual standard for global Settings and project Configure.
/// Every page stays mounted so Apply/Revert drafts survive sidebar navigation.
protocol SettingsNavigationItem: Identifiable, Hashable {
    var title: String { get }
    var symbol: String { get }
    var group: String { get }
}
extension SettingsPage: SettingsNavigationItem {}
extension ProjectConfigurePage: SettingsNavigationItem {}

struct SettingsPageShell<Page: SettingsNavigationItem, Content: View>: View {
    var title: String
    var searchPlaceholder: String
    var emptyTitle: String
    var clearSearchLabel: String
    var identifierPrefix: String
    var footer: String
    var groups: [String]
    var pages: [Page]
    var matchingPages: [Page]
    var subtitle: (Page) -> String
    @Binding var selected: Page
    @Binding var search: String
    @ViewBuilder var content: (Page) -> Content
    @Environment(\.appearanceTheme) private var theme
    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 18) {
                Text(title).font(.system(size: 21, weight: .semibold))
                HStack(spacing: 7) {
                    Image(systemName: "magnifyingglass").foregroundStyle(MLTheme.muted)
                    TextField(searchPlaceholder, text: $search).textFieldStyle(.plain)
                        .accessibilityIdentifier("\(identifierPrefix).search")
                    if !search.isEmpty {
                        Button { search = "" } label: { Image(systemName: "xmark.circle.fill") }
                            .buttonStyle(.plain).accessibilityLabel(clearSearchLabel)
                    }
                }.font(.system(size: 12)).padding(9).background(MLTheme.background)
                    .clipShape(Rectangle())
                    .overlay(Rectangle().stroke(MLTheme.rule))
                ThemedScrollView {
                    VStack(alignment: .leading, spacing: 5) {
                        ForEach(groups, id: \.self) { group in
                            let items = matchingPages.filter { $0.group == group }
                            if !items.isEmpty {
                                SectionLabel(title: group).padding(.top, 15).padding(.bottom, 7).padding(.leading, 8)
                                ForEach(items) { page in
                                    Button { selected = page } label: {
                                        HStack(spacing: 10) {
                                            Image(systemName: page.symbol).font(.system(size: 16)).frame(width: 20)
                                            Text(page.title).font(.system(size: 13))
                                            Spacer(minLength: 0)
                                        }.padding(.horizontal, 10).padding(.vertical, 11)
                                            .foregroundStyle(selected == page ? theme.accentForeground : Color.primary)
                                            .background(selected == page ? theme.accentColor : Color.clear)
                                            .clipShape(Rectangle())
                                    }.buttonStyle(.plain).accessibilityIdentifier("\(identifierPrefix).page.\(page.id)")
                                        .accessibilityAddTraits(selected == page ? [.isSelected] : [])
                                }
                            }
                        }
                        if matchingPages.isEmpty { Text(emptyTitle).font(.caption).foregroundStyle(MLTheme.muted).padding(8) }
                    }
                }
                Text(footer).font(.caption).foregroundStyle(MLTheme.muted)
            }.padding(18).frame(width: 224).background(MLTheme.panel)
            Divider()
            ZStack {
                ForEach(pages) { page in
                    ThemedScrollView {
                        VStack(alignment: .leading, spacing: 24) {
                            VStack(alignment: .leading, spacing: 8) {
                                Text(page.title).font(.system(size: 26, weight: .semibold))
                                Text(subtitle(page)).font(.system(size: 13)).foregroundStyle(MLTheme.muted)
                            }
                            Divider()
                            content(page)
                        }.padding(30).frame(maxWidth: .infinity, alignment: .leading)
                    }.opacity(selected == page ? 1 : 0).allowsHitTesting(selected == page)
                        .accessibilityHidden(selected != page)
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
        }.background(MLTheme.background)
    }
}
