import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ConfigurableActivationLearningStoreTests:XCTestCase {
    @MainActor func testAllThreeActualGuidedPrivateWorkersAndSavedReplay()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        _ = try XCTUnwrap(workspace.runtime)
        let lesson=TensorLearningStore(workspace:workspace)
        await verifyRuntimeForIntegration(workspace)
        func wait()async throws {
            let deadline=Date().addingTimeInterval(30)
            while workspace.isAssessingLearning,Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
            XCTAssertFalse(workspace.isAssessingLearning)
        }
        for task in ConfigurableActivationLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue),lesson.statusMessage)
            let graph=GraphDocument(name:"My local windows",nodes:[.init(id:"observations",type:"input",name:"Fields",x:0,y:0),.init(id:"kernel",type:task.componentID,name:"My window",x:200,y:0,config:task.configuration),.init(id:"results",type:"output",name:"New fields",x:400,y:0)],edges:[.init(id:"first",source:"observations",target:"kernel"),.init(id:"second",source:"kernel",target:"results")])
            XCTAssertTrue(lesson.drafts.update {$0.graph=graph})
            XCTAssertTrue(lesson.canSubmit,lesson.statusMessage)
            lesson.submit();try await wait()
            XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
            XCTAssertTrue(lesson.resultSaved);XCTAssertNotNil(lesson.formativeProjection)
            let id=try XCTUnwrap(workspace.activeProjectID),saved=workspace.graph
            XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Other work")))
            XCTAssertTrue(workspace.beginProject(saved,id:id))
            XCTAssertEqual(lesson.grade?.status,.passed);XCTAssertTrue(lesson.resultIsCurrent)
            XCTAssertTrue(lesson.setMode(.assessment));XCTAssertNil(lesson.formativeProjection)
            lesson.submit();try await wait()
            XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage);XCTAssertTrue(lesson.resultSaved)
            XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Private replay check")))
            XCTAssertTrue(workspace.beginProject(saved,id:id))
            XCTAssertEqual(lesson.grade?.status,.passed);XCTAssertTrue(lesson.resultSaved)
            XCTAssertNil(lesson.formativeProjection)
            XCTAssertTrue(lesson.setMode(.guided))
            XCTAssertNil(lesson.grade,"Changing mode intentionally clears its current result")
            XCTAssertNil(lesson.formativeProjection,"An assessment-origin result stays private after changing mode")
            XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Separate work")))
            XCTAssertTrue(workspace.beginProject(saved,id:id))
            XCTAssertNil(lesson.grade,"An assessment attempt is not presented as a guided result")
            XCTAssertFalse(lesson.resultSaved)
            XCTAssertTrue(lesson.statusMessage.contains("another learning mode"),lesson.statusMessage)
            XCTAssertNil(lesson.formativeProjection)
        }
    }
}
