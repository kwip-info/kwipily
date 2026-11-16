import SwiftUI
import KwipMLCore

/// Reading navigation only. Opening this link never launches lesson work or records progress.
struct ComponentLessonLink: View {
    private let entryID: String?
    init(entryID: String) { self.entryID = entryID }
    init(type: String, configuration: [String: JSONValue] = [:]) {
        self.entryID = ComponentLessonDestination.entryID(type: type, configuration: configuration)
    }
    var body: some View {
        if let entryID, ComponentLessonDestination.isAvailable(entryID: entryID) {
            Button { ApplicationNavigationWindows.shared.showLearn(entryID: entryID) } label: {
                Label(entryID == "module-composition" ? "Read about Blocks" : entryID == "guided" ? "Read the canvas guide" : "Read the lesson", systemImage: "book")
            }.buttonStyle(.link)
                .accessibilityIdentifier("component.lesson." + entryID)
        }
    }
}
