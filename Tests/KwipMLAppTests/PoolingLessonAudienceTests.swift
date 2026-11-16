import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PoolingLessonAudienceTests:XCTestCase {
    func testAllFourHavePracticalControlsAndPartialWindowExplanation() {
        for task in PoolingLearningTask.allCases {
            let required=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"")
            for term in ["**Configure**","Select","Check construction","automatically","partial"] {XCTAssertTrue(required.contains(term),term)}
            let guide=LessonAudienceGuide.make(operation:task.rawValue)
            XCTAssertTrue(Set(guide.terms.map {$0.name.lowercased()}).isSuperset(of:["window","channel","stride","padding","partial window","gradient"]))
        }
        XCTAssertTrue(PoolingLearningTask.averageOne.instructions.contains("Include padding in average divisor"))
        for task in PoolingLearningTask.allCases where !task.isAverage {XCTAssertTrue(task.instructions.contains("Expose winning indices"))}
    }
    @MainActor func testMaxIndicesPortIsIntegerAndUnavailableWhenDisabled()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:PoolingLearningTask.maxOne.rawValue))
        let canvas=try XCTUnwrap(lesson.canvas)
        canvas.addNode(type:"max_pool1d",x:0,y:0)
        let node=try XCTUnwrap(canvas.graph.nodes.first)
        XCTAssertTrue(canvas.configure(nodeID:node.id,values:PoolingLearningTask.maxOne.configuration))
        XCTAssertTrue(canvas.canvasPortDescription(nodeID:node.id,port:"indices").contains("int64"))
        var disabled=PoolingLearningTask.maxOne.configuration;disabled["return_indices"]=false
        XCTAssertTrue(canvas.configure(nodeID:node.id,values:disabled))
        XCTAssertTrue(canvas.canvasPortDescription(nodeID:node.id,port:"indices").localizedCaseInsensitiveContains("unavailable"))
    }
}
