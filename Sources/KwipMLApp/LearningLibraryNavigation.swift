import Foundation
import Combine

/// Application-level reading state. A lesson link changes the library view only;
/// it never creates a project, advances a course or starts a computation.
@MainActor final class LearningLibraryNavigation: ObservableObject {
    @Published var section = "Overview"
    @Published var search = ""
    @Published var typeFilter = "All types"
    @Published var frameworkFilter = "All frameworks"
    @Published private(set) var directEntryID: String?
    private(set) var browseAnchorID: String?
    func open(search query: String) {
        directEntryID = nil
        section = "Library"
        search = query.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    @discardableResult func open(entryID: String) -> Bool {
        guard ComponentLessonDestination.isAvailable(entryID: entryID) else { return false }
        section = "Library"
        search = ""
        browseAnchorID = nil
        directEntryID = entryID
        return true
    }
    /// Reading is a drilldown over the current browsing context, not a new search.
    @discardableResult func openFromBrowse(entryID: String) -> Bool {
        guard ComponentLessonDestination.isAvailable(entryID: entryID) else { return false }
        browseAnchorID = entryID
        directEntryID = entryID
        return true
    }
    func browseLibrary() { directEntryID = nil }
    func showSection(_ next: String) {
        directEntryID = nil; browseAnchorID = nil; section = next == "Learning paths" ? "Overview" : next; search = ""
    }
}
