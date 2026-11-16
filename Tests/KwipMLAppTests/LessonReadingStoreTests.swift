import XCTest
@testable import KwipMLApp

@MainActor final class LessonReadingStoreTests: XCTestCase {
    func testReadingResumeSavingAndContentRevisionsAreIndependent() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        var content=LessonContent(entryID:"example",title:"Example",kind:"Example",steps:[.stepFixture("first"),.stepFixture("second")])
        let store=LessonReadingStore(directory:root)
        XCTAssertEqual(store.status(content),.new)
        XCTAssertFalse(store.visit(2,content:content))
        XCTAssertTrue(store.visit(0,content:content));XCTAssertEqual(store.status(content),.inProgress)
        XCTAssertNil(store.record("example").savedProjectID)
        XCTAssertTrue(store.visit(1,content:content));XCTAssertEqual(store.status(content),.viewed)
        XCTAssertTrue(store.saved(entryID:"example",projectID:"project-a"))
        let reopened=LessonReadingStore(directory:root)
        XCTAssertEqual(reopened.position(content),1)
        XCTAssertEqual(reopened.entry(projectID:"project-a"),"example")
        XCTAssertEqual(reopened.record("example").savedProjectID,"project-a")
        content.steps[1].blocks[0].text="A revised explanation"
        XCTAssertEqual(reopened.status(content),.inProgress)
        XCTAssertEqual(reopened.position(content),1)
        XCTAssertTrue(reopened.visit(1,content:content));XCTAssertEqual(reopened.status(content),.viewed)
    }
    func testUnsavePreservesReadingAndProjectAssociationAndCanBeSavedAgain() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let content=LessonContent(entryID:"example",title:"Example",kind:"Example",steps:[.stepFixture("first")])
        let store=LessonReadingStore(directory:root)
        XCTAssertTrue(store.visit(0,content:content))
        XCTAssertTrue(store.saved(entryID:"example",projectID:"project-a"))
        XCTAssertTrue(store.unsave(projectID:"project-a"))
        let reopened=LessonReadingStore(directory:root)
        XCTAssertTrue(reopened.isUnsaved(projectID:"project-a"))
        XCTAssertNil(reopened.record("example").savedProjectID)
        XCTAssertEqual(reopened.entry(projectID:"project-a"),"example")
        XCTAssertEqual(reopened.status(content),.viewed)
        XCTAssertTrue(reopened.saved(entryID:"example",projectID:"project-a"))
        XCTAssertFalse(reopened.isUnsaved(projectID:"project-a"))
        XCTAssertEqual(reopened.record("example").savedProjectID,"project-a")
        XCTAssertTrue(reopened.unsave(projectID:"legacy-construction"))
        XCTAssertTrue(LessonReadingStore(directory:root).isUnsaved(projectID:"legacy-construction"))
    }
    func testDifferentAssessmentInOneFamilyStartsAtIntroductionWithoutBorrowingViewedStatus()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=LessonReadingStore(directory:root)
        let first=LessonContent(entryID:"affine",title:"First task",kind:"Construction",steps:[.stepFixture("goal"),.stepFixture("inspect")],readingScopeID:"assessment.first")
        let second=LessonContent(entryID:"affine",title:"Second task",kind:"Construction",steps:first.steps,readingScopeID:"assessment.second")
        XCTAssertTrue(store.visit(0,content:first));XCTAssertTrue(store.visit(1,content:first))
        XCTAssertTrue(store.saved(entryID:"affine",projectID:"old-attempt"))
        XCTAssertEqual(store.position(first),1)
        XCTAssertEqual(store.status(first),.viewed)
        XCTAssertEqual(store.position(second),0)
        XCTAssertEqual(store.status(second),.new)
        XCTAssertEqual(store.record("affine").content,first)
        XCTAssertEqual(store.entry(projectID:"old-attempt"),"affine")
        XCTAssertEqual(store.record("affine").savedProjectID,"old-attempt")
        XCTAssertTrue(store.visit(0,content:second))
        let reopened=LessonReadingStore(directory:root)
        XCTAssertEqual(reopened.position(first),1)
        XCTAssertEqual(reopened.status(first),.viewed)
        XCTAssertEqual(reopened.position(second),0)
        XCTAssertEqual(reopened.status(second),.inProgress)
        XCTAssertEqual(reopened.record("affine").savedProjectID,"old-attempt")
        XCTAssertEqual(reopened.entry(projectID:"old-attempt"),"affine")
    }
    func testLegacyReadingIsRetainedWithoutGrantingAnotherTaskItsPosition()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=LessonReadingStore(directory:root)
        let old=LessonContent(entryID:"affine",title:"Older task",kind:"Construction",steps:[.stepFixture("goal"),.stepFixture("inspect")])
        XCTAssertTrue(store.visit(1,content:old))
        var current=old;current.readingScopeID="assessment.specific"
        XCTAssertEqual(store.position(current),0)
        XCTAssertTrue(store.visit(0,content:current))
        XCTAssertEqual(store.record("legacy-reading/affine").content,old)
        XCTAssertEqual(store.record("legacy-reading/affine").position,"inspect")
        let data=try JSONEncoder().encode(current)
        XCTAssertEqual(try JSONDecoder().decode(LessonContent.self,from:data),current)
        XCTAssertNil(try JSONDecoder().decode(LessonContent.self,from:JSONEncoder().encode(old)).readingScopeID)
    }
    func testCorruptProgressIsNotOverwrittenOrPresentedAsSaved() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let file=root.appendingPathComponent("learning/reading.json")
        try FileManager.default.createDirectory(at:file.deletingLastPathComponent(),withIntermediateDirectories:true)
        let original=Data("not valid progress".utf8);try original.write(to:file)
        let store=LessonReadingStore(directory:root)
        XCTAssertNotNil(store.error)
        XCTAssertFalse(store.saved(entryID:"example",projectID:"p"))
        XCTAssertFalse(store.unsave(projectID:"p"))
        XCTAssertEqual(try Data(contentsOf:file),original)
    }
}
private extension LessonContentStep {
    static func stepFixture(_ id:String)->Self {LessonContent.step(id,id,"Read this idea")}
}
