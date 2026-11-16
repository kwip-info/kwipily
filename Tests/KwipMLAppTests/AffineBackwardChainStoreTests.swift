import XCTest
import KwipMLCore
@testable import KwipMLApp

/// Derives an explicit reverse chain from the public forward matrices, never a hidden oracle.
final class AffineBackwardChainStoreTests:XCTestCase {
    @MainActor func testAnalyticalBackwardConstructionFailureRepairReceiptAndModePrivacy()async throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath:root.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development worker unavailable")}
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        let lesson=AffineLearningStore(workspace:workspace)
        lesson.start(identity:.backwardChain,mode:.guided)
        let projectID=try XCTUnwrap(workspace.activeProjectID)
        XCTAssertEqual(lesson.lessonTask?.identity,.backwardChain)
        XCTAssertEqual(lesson.lessonTask?.identity.taskSeed,nil)
        XCTAssertEqual(workspace.graph.nodes.map(\.type),["input","output"],"The lesson must remain unsolved at entry.")
        XCTAssertTrue(workspace.graph.edges.isEmpty)
        XCTAssertEqual(workspace.graph.nodes.first(where:{$0.id=="input"})?.name,"Upstream gradient")
        XCTAssertEqual(workspace.graph.nodes.first(where:{$0.id=="output"})?.name,"Input gradient")
        let forward=LearningExamples.affineBackwardReference.forward
        let first=transpose(forward.second.weight),second=transpose(forward.first.weight)
        workspace.edit { document in
            document=GraphDocument(nodes:[
                GraphNode(id:"input",type:"input",config:["features":.integer(Int64(first.shape[1]))]),
                linear("reverse2",weight:first),linear("reverse1",weight:second),GraphNode(id:"output",type:"output")
            ],edges:[GraphEdge(source:"input",target:"reverse2"),GraphEdge(source:"reverse2",target:"reverse1"),GraphEdge(source:"reverse1",target:"output")])
        }
        for (id,weight) in [("reverse2",first),("reverse1",second)] {enter(weight,at:id,lesson:lesson,workspace:workspace)}
        lesson.setNotes("The upstream row is multiplied by W2 then W1. Forward offsets do not add a gradient offset.")
        lesson.revealHint("backward-order")
        lesson.submit();try await finish(workspace)
        XCTAssertEqual(lesson.latestGrade?.status,.passed,lesson.statusMessage)
        XCTAssertFalse(lesson.hasCurrentCompletion,"Guided checks cannot award assessment completion.")
        XCTAssertEqual(lesson.formativeProjection?.fixtures.count,4)
        XCTAssertEqual(lesson.latestAttempt?.objectiveID,"matrices.affine-backward-chain")
        XCTAssertEqual(lesson.latestAttempt?.assessmentVersion,ContractVersion())

        // A constant additive offset is not the VJP, even though the forward maps have biases.
        workspace.selectNode("reverse1")
        workspace.edit { graph in graph.nodes[graph.nodes.firstIndex{$0.id=="reverse1"}!].config["bias"]=true }
        lesson.resetParameters(nodeID:"reverse1");enter(second,at:"reverse1",lesson:lesson,workspace:workspace,reset:false)
        XCTAssertTrue(lesson.setBias(nodeID:"reverse1",index:0,value:1))
        XCTAssertFalse(lesson.latestResultIsCurrent);XCTAssertNil(lesson.formativeProjection)
        lesson.submit();try await finish(workspace)
        XCTAssertEqual(lesson.latestGrade?.status,.failed,lesson.statusMessage)
        XCTAssertFalse(lesson.hasCurrentCompletion)

        workspace.edit { graph in graph.nodes[graph.nodes.firstIndex{$0.id=="reverse1"}!].config["bias"]=false }
        lesson.resetParameters(nodeID:"reverse1");enter(second,at:"reverse1",lesson:lesson,workspace:workspace,reset:false)
        lesson.setMode(.assessment);lesson.submit();try await finish(workspace)
        XCTAssertEqual(lesson.latestGrade?.status,.passed,lesson.statusMessage)
        XCTAssertTrue(lesson.resultSaved);XCTAssertTrue(lesson.hasCurrentCompletion)
        let attempt=try XCTUnwrap(lesson.latestAttempt)
        let receipt=try JSONDecoder().decode(AffineLearningReceipt.self,from:Data(contentsOf:directory.appendingPathComponent("learning/attempts/\(attempt.id).json")))
        XCTAssertEqual(receipt.lessonIdentity,.backwardChain)
        XCTAssertEqual(receipt.runtime.backend,"pytorch")
        XCTAssertEqual(receipt.runtime.capabilities["linear"],ContractVersion(1,1,0))
        XCTAssertEqual(receipt.submission.schemaVersion,2)
        XCTAssertNil(lesson.formativeProjection)
        lesson.setMode(.practice)
        XCTAssertNil(lesson.formativeProjection,"Switching modes must not reveal assessment fixtures.")
        XCTAssertTrue(lesson.save())
        let savedNotes=lesson.draft?.notes
        let reopened=AffineLearningStore(workspace:workspace);reopened.reopenCurrentProject()
        XCTAssertEqual(reopened.draft?.projectID,projectID)
        XCTAssertEqual(reopened.lessonTask?.identity,.backwardChain)
        XCTAssertEqual(reopened.latestAttempt?.id,attempt.id)
        XCTAssertEqual(reopened.draft?.notes,savedNotes)
        XCTAssertTrue(reopened.draft?.hintIDs.contains("backward-order") == true)
        XCTAssertTrue(reopened.latestResultIsCurrent);XCTAssertTrue(reopened.hasCurrentCompletion)
        XCTAssertNil(reopened.formativeProjection)
        reopened.submit();try await finish(workspace)
        XCTAssertEqual(reopened.latestGrade?.status,.passed,reopened.statusMessage)
        XCTAssertEqual(reopened.formativeProjection?.fixtures.count,4)
        reopened.start(identity:.composition,mode:.guided)
        XCTAssertEqual(reopened.lessonTask?.identity,.composition)
        XCTAssertFalse(reopened.hasCurrentCompletion,"Backward-chain credit must not award forward-composition completion.")
    }
    private func transpose(_ tensor:LearningTensor)->LearningTensor {
        let rows=tensor.shape[0],columns=tensor.shape[1]
        return LearningTensor(shape:[columns,rows],values:(0..<columns).flatMap { column in (0..<rows).map { row in tensor.values[row*columns+column] } })
    }
    private func linear(_ id:String,weight:LearningTensor)->GraphNode {
        GraphNode(id:id,type:"linear",config:["in_features":.integer(Int64(weight.shape[1])),"out_features":.integer(Int64(weight.shape[0])),"bias":false])
    }
    @MainActor private func enter(_ weight:LearningTensor,at id:String,lesson:AffineLearningStore,workspace:WorkspaceStore,reset:Bool=true) {
        workspace.selectNode(id)
        if reset {lesson.resetParameters(nodeID:id)}
        for (index,value) in weight.values.enumerated() {
            XCTAssertTrue(lesson.setWeight(nodeID:id,row:index/weight.shape[1],column:index%weight.shape[1],value:value))
        }
    }
    @MainActor private func finish(_ workspace:WorkspaceStore)async throws {
        let deadline=Date().addingTimeInterval(25)
        while workspace.isAssessingLearning && Date()<deadline {try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertFalse(workspace.isAssessingLearning)
    }
}
