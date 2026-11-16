import XCTest
import KwipMLCore
@testable import KwipMLApp

final class EmbeddingBagLessonLifecycleTests:XCTestCase {
    @MainActor private func build(_ task:EmbeddingBagLearningTask,in lesson:TensorLearningStore)throws->TensorLearningCanvasAdapter {
        XCTAssertTrue(lesson.start(operation:task.rawValue,mode:.guided))
        let canvas=try XCTUnwrap(lesson.canvas)
        XCTAssertEqual(Set(canvas.registry.components.map(\.id)),["bag_input","embedding_bag","output"])
        XCTAssertNil(canvas.registry.component(type:"input"))
        for type in ["bag_input","embedding_bag","output"] {canvas.addNode(type:type,x:0,y:0)}
        let source=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "bag_input"})
        let combine=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "embedding_bag"})
        let output=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "output"})
        XCTAssertEqual(canvas.graph.nodes.count,3)
        XCTAssertFalse(canvas.validation.isValid,"An incomplete graph remains editable but cannot be submitted")
        XCTAssertTrue(canvas.configure(nodeID:source.id,values:try task.sourceConfiguration.values))
        XCTAssertTrue(canvas.configure(nodeID:combine.id,values:try task.configuration.values))
        for role in try task.sourceConfiguration.outputPorts {
            XCTAssertTrue(canvas.canConnect(source:source.id,sourcePort:role,target:combine,port:role))
            canvas.connect(source:source.id,sourcePort:role,target:combine.id,targetPort:role)
        }
        canvas.connect(source:combine.id,sourcePort:"out",target:output.id,targetPort:"in")
        XCTAssertTrue(canvas.validation.isValid,canvas.validation.issues.map(\.message).joined(separator:"; "))
        XCTAssertEqual(Set(canvas.graph.edges.filter {$0.target == combine.id}.map(\.targetPort)),Set(try task.configuration.inputPorts))
        return canvas
    }
    @MainActor private func check(_ lesson:TensorLearningStore,workspace:WorkspaceStore,expected:LearningGradeStatus = .passed)async throws {
        XCTAssertTrue(lesson.canSubmit,lesson.runtimeReadinessError ?? lesson.statusMessage)
        lesson.submit()
        let deadline=Date().addingTimeInterval(25)
        while workspace.isAssessingLearning && Date()<deadline {try await Task.sleep(nanoseconds:20_000_000)}
        XCTAssertFalse(workspace.isAssessingLearning,"Bag lesson exceeded its bounded worker deadline")
        XCTAssertEqual(lesson.grade?.status,expected,lesson.statusMessage)
        XCTAssertTrue(lesson.resultSaved,lesson.statusMessage)
        XCTAssertTrue(lesson.resultIsCurrent)
    }
    @MainActor func testAllSixActualManagedCanvasLessonsGradeReopenAndKeepPrivateOriginHidden()async throws {
        guard let path=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable development runtime for actual managed bag lesson lifecycle checks.")}
        let repository=URL(fileURLWithPath:path)
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        // One genuine discovery for this workspace; no injected capability fixtures.
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.embeddingBagRuntimeCapabilities.isComplete)
        var saved:[(task:EmbeddingBagLearningTask,id:String,graph:GraphDocument)]=[]
        for task in EmbeddingBagLearningTask.allCases {
            let canvas=try build(task,in:lesson)
            XCTAssertNil(lesson.runtimeReadinessError)
            try await check(lesson,workspace:workspace)
            let formative=try XCTUnwrap(lesson.formativeProjection)
            XCTAssertEqual(formative.fixtures.count,4)
            let config=try task.configuration
            for fixture in formative.fixtures {
                XCTAssertEqual(fixture.inputLabel,"Category IDs by observation group")
                XCTAssertEqual(fixture.sourceLayout,config.layout)
                XCTAssertEqual(fixture.sourceLengths != nil,config.layout == "packed")
                XCTAssertEqual(fixture.sourceWeights != nil,config.usesPerSampleWeights)
                XCTAssertEqual(fixture.measured["sourceWeightsGradient"] != nil,config.usesPerSampleWeights)
                XCTAssertNil(fixture.measured["inputGradient"])
                XCTAssertNil(fixture.measured["indicesGradient"])
                XCTAssertNil(fixture.measured["offsetsGradient"])
            }
            let combine=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "embedding_bag"})
            var wrong=combine.config
            if task == .paddingNorm {wrong["max_norm"] = .null} else {wrong["padding_idx"] = .integer(0)}
            XCTAssertTrue(canvas.configure(nodeID:combine.id,values:wrong))
            XCTAssertTrue(canvas.validation.isValid,"Wrong requested behavior is still a valid configurable model")
            XCTAssertFalse(lesson.resultIsCurrent)
            XCTAssertNil(lesson.formativeProjection,"Changed construction cannot reuse old results")
            XCTAssertEqual(canvas.previewStatus(for:combine.id),"Stale")
            canvas.undo()
            XCTAssertTrue(lesson.resultIsCurrent)
            XCTAssertNotNil(lesson.formativeProjection)
            if task == .fixedSum {
                // A real valid-but-wrong model must fail independent grading, not only stale display.
                XCTAssertTrue(canvas.configure(nodeID:combine.id,values:wrong))
                try await check(lesson,workspace:workspace,expected:.failed)
                XCTAssertTrue(canvas.configure(nodeID:combine.id,values:try task.configuration.values))
                XCTAssertFalse(lesson.resultIsCurrent)
            }
            XCTAssertTrue(lesson.setMode(.assessment))
            XCTAssertNil(lesson.formativeProjection)
            try await check(lesson,workspace:workspace)
            XCTAssertNil(lesson.formativeProjection)
            XCTAssertTrue(canvas.previewStatus(for:combine.id).contains("withheld"))
            XCTAssertTrue(lesson.save())
            saved.append((task,try XCTUnwrap(workspace.activeProjectID),workspace.graph))
        }
        // New store/session instances replay disk receipts; no warm in-memory receipt is reused.
        let cold=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let coldLesson=TensorLearningStore(workspace:cold)
        for record in saved {
            XCTAssertTrue(cold.beginProject(record.graph,id:record.id))
            XCTAssertEqual(coldLesson.draft?.operation,record.task.rawValue)
            XCTAssertEqual(coldLesson.draft?.mode,.assessment)
            XCTAssertEqual(coldLesson.grade?.status,.passed,coldLesson.statusMessage)
            XCTAssertTrue(coldLesson.resultSaved)
            XCTAssertTrue(coldLesson.resultIsCurrent)
            XCTAssertNil(coldLesson.formativeProjection)
            XCTAssertNotNil(coldLesson.runtimeReadinessError,"Receipt replay does not grant discovery to a cold workspace")
            XCTAssertTrue(coldLesson.setMode(.guided))
            XCTAssertNil(coldLesson.grade)
            XCTAssertNil(coldLesson.formativeProjection,"A private-origin result cannot become formative through a mode switch")
        }
    }
}
