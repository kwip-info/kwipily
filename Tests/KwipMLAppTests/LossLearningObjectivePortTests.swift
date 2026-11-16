import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LossLearningObjectivePortTests:XCTestCase {
    @MainActor func testEightLessonsDistinguishRecipeInputFromCombinedScoreOutputWithSamePortName()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        for task in LossLearningTask.allCases {
            let session=TensorLearningDraftSession(repository:.init(directory:root))
            XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided))
            XCTAssertTrue(session.update {$0.graph=LossLearningCanvas.preparedGraph(task:task)})
            let canvas=TensorLearningCanvasAdapter(session:session)
            let recipe=canvas.canvasPortDescriptor(nodeID:"objective",port:"out",direction:.output)
            let required=canvas.canvasPortDescriptor(nodeID:"check",port:"objective",direction:.input)
            XCTAssertEqual(recipe.kind,.objectiveRecipe,task.rawValue)
            XCTAssertEqual(required.kind,.objectiveRecipe,task.rawValue)
            XCTAssertTrue(required.accessibilityDescription.contains("No tensor dimensions"))
            let score=canvas.canvasPortDescriptor(nodeID:"check",port:"objective",direction:.output)
            let consumer=canvas.canvasPortDescriptor(nodeID:"result-objective",port:"in",direction:.input)
            for descriptor in [score,consumer] {
                XCTAssertEqual(descriptor.kind,.tensor,task.rawValue)
                XCTAssertTrue(descriptor.declared.contains("[1]"),descriptor.accessibilityDescription)
                XCTAssertTrue(descriptor.declared.contains("float64"),descriptor.accessibilityDescription)
                XCTAssertTrue(descriptor.declared.contains("One combined score · no observation direction"),descriptor.accessibilityDescription)
                XCTAssertFalse(descriptor.accessibilityDescription.contains("No tensor dimensions"))
                XCTAssertTrue(descriptor.measured.isEmpty,"A declared shape is not measured evidence")
            }
            XCTAssertEqual(score.declared,consumer.declared,task.rawValue)
            for port in ["raw","inputInfluence"] {
                let value=canvas.canvasPortDescriptor(nodeID:"check",port:port,direction:.output)
                XCTAssertEqual(value.kind,.tensor)
                XCTAssertTrue(value.declared.contains("float64"))
                XCTAssertFalse(value.accessibilityDescription.contains("No tensor dimensions"))
            }
            XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:"check",port:"missing",direction:.output).kind,.unknown)
        }
    }
}
