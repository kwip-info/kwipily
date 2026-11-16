import XCTest
@testable import KwipMLApp

@MainActor final class LearningLibraryPresentationTests: XCTestCase {
    func testCompactDiscoveryPreservesEveryPreviouslyDisplayedStableIdentityExactlyOnce() {
        let expected = LearnCatalogContent.items.map(\.id) + LearnCatalogContent.displayedComponentExampleEntries.map(\.componentID)
        let actual = LearningLibraryPresentation.items.map(\.id)
        XCTAssertEqual(actual, expected)
        XCTAssertEqual(Set(actual).count, actual.count)
        XCTAssertTrue(actual.allSatisfy { ComponentLessonDestination.isAvailable(entryID: $0) })
        let grouped = LearningLibraryPresentation.topics.flatMap { topic in
            LearningLibraryPresentation.matching(section: topic, search: "").map(\.id)
        }
        XCTAssertEqual(Set(grouped), Set(actual))
        XCTAssertEqual(grouped.count, actual.count)
    }
    func testTopicTypeFrameworkAndSearchAreIntersectedWithoutChangingCatalog() {
        let all = LearningLibraryPresentation.items
        let filtered = LearningLibraryPresentation.matching(section: "Layers & activations", search: "convolution", type: "Example", framework: "PyTorch")
        XCTAssertFalse(filtered.isEmpty)
        XCTAssertTrue(filtered.allSatisfy { $0.category == "Layers & activations" && $0.kind == "Example" && $0.backend == "PyTorch" && $0.matches(section: "Library", search: "convolution") })
        XCTAssertEqual(LearningLibraryPresentation.matching(section: "Library", search: "", type: "All types", framework: "All frameworks").map(\.id), all.map(\.id))
        XCTAssertTrue(LearningLibraryPresentation.matching(section: "Library", search: "no-such-learning-entry-xyz").isEmpty)
        for type in LearningLibraryPresentation.types {
            XCTAssertEqual(LearningLibraryPresentation.matching(section: "Library", search: "", type: type).map(\.id), all.filter { $0.kind == type }.map(\.id))
        }
        for framework in LearningLibraryPresentation.frameworks {
            XCTAssertEqual(LearningLibraryPresentation.matching(section: "Library", search: "", framework: framework).map(\.id), all.filter { $0.backend == framework }.map(\.id))
        }
    }
    func testReadingAndBackPreserveBrowseSearchTopicFiltersAndRowAnchor() {
        let navigation = LearningLibraryNavigation()
        navigation.section = "Layers & activations"
        navigation.search = "convolution"
        navigation.typeFilter = "Example"
        navigation.frameworkFilter = "PyTorch"
        XCTAssertTrue(navigation.openFromBrowse(entryID: "conv2d"))
        XCTAssertEqual(navigation.directEntryID, "conv2d")
        navigation.browseLibrary()
        XCTAssertNil(navigation.directEntryID)
        XCTAssertEqual(navigation.section, "Layers & activations")
        XCTAssertEqual(navigation.search, "convolution")
        XCTAssertEqual(navigation.typeFilter, "Example")
        XCTAssertEqual(navigation.frameworkFilter, "PyTorch")
        XCTAssertEqual(navigation.browseAnchorID, "conv2d")
        XCTAssertFalse(navigation.openFromBrowse(entryID: "missing-entry"))
        XCTAssertNil(navigation.directEntryID)
        navigation.showSection("Saved lessons")
        XCTAssertEqual(navigation.section, "Saved lessons")
        XCTAssertEqual(navigation.search, "")
        XCTAssertNil(navigation.browseAnchorID)
    }
    func testSuggestionsAreExistingAuthoredStopsAndDoNotInventProgress() {
        XCTAssertEqual(LearningLibraryPresentation.suggestions.count, 3)
        XCTAssertEqual(LearningJourney.all.count, 6)
        for suggestion in LearningLibraryPresentation.suggestions {
            XCTAssertEqual(suggestion.step.entryID, suggestion.journey.steps.first?.entryID)
            XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID: suggestion.step.entryID))
            XCTAssertFalse(suggestion.step.advice.isEmpty)
            XCTAssertEqual(LearningLibraryPresentation.firstItem(in: suggestion.journey)?.id, suggestion.step.entryID)
        }
        for journey in LearningJourney.all {
            XCTAssertEqual(LearningLibraryPresentation.firstItem(in: journey)?.id, journey.steps.first?.entryID)
        }
        for item in LearningLibraryPresentation.items {
            XCTAssertFalse(LearningLibraryPresentation.purpose(item).isEmpty)
            XCTAssertTrue(LearningLibraryPresentation.topics.contains(item.category))
        }
    }
}
