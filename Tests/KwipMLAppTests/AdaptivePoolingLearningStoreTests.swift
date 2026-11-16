import XCTest
import KwipMLCore
@testable import KwipMLApp

final class AdaptivePoolingLearningStoreTests:XCTestCase {
    @MainActor func testEveryTaskWorkerColdReceiptAndRepresentativeModes()async throws {
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
        for task in AdaptivePoolingLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue),lesson.statusMessage)
            let graph=GraphDocument(name:"My local windows",nodes:[.init(id:"observations",type:"input",name:"Fields",x:0,y:0),.init(id:"kernel",type:task.componentID,name:"My window",x:200,y:0,config:task.configuration),.init(id:"results",type:"output",name:"New fields",x:400,y:0)],edges:[.init(id:"first",source:"observations",target:"kernel"),.init(id:"second",source:"kernel",target:"results")])
            XCTAssertTrue(lesson.drafts.update {$0.graph=graph})
            XCTAssertTrue(lesson.canSubmit,lesson.statusMessage)
            lesson.submit();try await wait()
            XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
            XCTAssertTrue(lesson.resultSaved);XCTAssertNotNil(lesson.formativeProjection)
            let id=try XCTUnwrap(workspace.activeProjectID),saved=workspace.graph
            // Exercise mode-specific receipt admission using this real measured evidence.
            // These projections are metadata checks, not additional worker submissions.
            let receipts=TensorLearningReceiptRepository(directory:storage.appendingPathComponent("learning/projects"))
            let measured=try XCTUnwrap(receipts.latest(projectID:id))
            let plan=try measured.verifiedReplay(expectedProjectID:id)
            for mode in [LearningMode.guided,.practice,.assessment] {
                let receipt=try TensorLearningReceipt(projectID:id,operation:task.rawValue,mode:mode,
                    submissionData:measured.submissionData,plan:plan,evidence:measured.evidence)
                XCTAssertEqual(receipt.formativeProjection(expectedProjectID:id,currentSubmission:measured.submissionData,currentMode:mode) == nil,mode == .assessment)
                if mode == .assessment {XCTAssertNil(receipt.formativeProjection(expectedProjectID:id,currentSubmission:measured.submissionData,currentMode:.guided))}
            }
            XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Other work")))
            XCTAssertTrue(workspace.beginProject(saved,id:id))
            XCTAssertEqual(lesson.grade?.status,.passed);XCTAssertTrue(lesson.resultIsCurrent)
            if task == .averageTwo {
                XCTAssertTrue(lesson.setMode(.practice))
                lesson.submit();try await wait()
                XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage);XCTAssertTrue(lesson.resultSaved)
                XCTAssertNotNil(lesson.formativeProjection)
                XCTAssertTrue(lesson.setMode(.assessment));XCTAssertNil(lesson.formativeProjection)
                lesson.submit();try await wait()
                XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage);XCTAssertTrue(lesson.resultSaved)
                XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Private replay check")))
                XCTAssertTrue(workspace.beginProject(saved,id:id))
                XCTAssertEqual(lesson.grade?.status,.passed);XCTAssertTrue(lesson.resultSaved)
                XCTAssertNil(lesson.formativeProjection)
                let coldWorkspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
                let coldLesson=TensorLearningStore(workspace:coldWorkspace)
                XCTAssertTrue(coldWorkspace.beginProject(saved,id:id))
                XCTAssertEqual(coldLesson.grade?.status,.passed,coldLesson.statusMessage)
                XCTAssertTrue(coldLesson.resultSaved)
                XCTAssertTrue(coldLesson.resultIsCurrent)
                XCTAssertNil(coldLesson.formativeProjection,"Cold private results must not expose fixture tables")
                XCTAssertTrue(lesson.setMode(.guided))
                XCTAssertNil(lesson.grade,"Changing mode intentionally clears its current result")
                XCTAssertNil(lesson.formativeProjection,"An assessment-origin result stays private after changing mode")
                XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Separate work")))
                XCTAssertTrue(workspace.beginProject(saved,id:id))
                XCTAssertNil(lesson.grade,"An assessment attempt is not presented as a guided result")
                XCTAssertFalse(lesson.resultSaved)
                XCTAssertTrue(lesson.statusMessage.contains("another learning mode"),lesson.statusMessage)
                XCTAssertNil(lesson.formativeProjection)
            } else {
                let cold=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
                let coldLesson=TensorLearningStore(workspace:cold)
                XCTAssertTrue(cold.beginProject(saved,id:id))
                XCTAssertEqual(coldLesson.grade?.status,.passed,coldLesson.statusMessage)
                XCTAssertTrue(coldLesson.resultSaved);XCTAssertTrue(coldLesson.resultIsCurrent)
                XCTAssertNotNil(coldLesson.formativeProjection)
            }
            let savedDrafts=TensorLearningDraftRepository(directory:storage.appendingPathComponent("learning/projects"))
            for mode in [LearningMode.practice,.assessment,.guided] {
                XCTAssertTrue(lesson.setMode(mode));XCTAssertTrue(lesson.save())
                XCTAssertNil(lesson.grade);XCTAssertNil(lesson.formativeProjection);XCTAssertFalse(lesson.resultSaved)
                XCTAssertEqual(try savedDrafts.load(projectID:id).draft.mode,mode)
                if task == .averageTwo {
                    let reopened=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
                    let reopenedLesson=TensorLearningStore(workspace:reopened)
                    XCTAssertTrue(reopened.beginProject(saved,id:id));XCTAssertEqual(reopenedLesson.draft?.mode,mode)
                    if mode == .assessment {XCTAssertNil(reopenedLesson.formativeProjection)}
                }
            }
        }
    }
    @MainActor func testValidWrongAverageDivisorIsMeasuredFailedThenRepaired()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        await verifyRuntimeForIntegration(workspace)
        let task=AdaptivePoolingLearningTask.averageTwo
        XCTAssertTrue(lesson.start(operation:task.rawValue),lesson.statusMessage)
        var wrong=task.configuration;wrong["divisor_override"]=1
        let graph=GraphDocument(name:"Boundary averages",nodes:[.init(id:"x",type:"input",name:"Measurements",x:0,y:0),.init(id:"p",type:task.componentID,name:"My regions",x:200,y:0,config:wrong),.init(id:"y",type:"output",name:"Averages",x:400,y:0)],edges:[.init(id:"first",source:"x",target:"p"),.init(id:"second",source:"p",target:"y")])
        XCTAssertTrue(lesson.drafts.update {$0.graph=graph})
        XCTAssertTrue(lesson.canSubmit,"A valid custom divisor must be measured, not rejected as a malformed graph")
        func wait()async throws {
            let deadline=Date().addingTimeInterval(30)
            while workspace.isAssessingLearning,Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
            XCTAssertFalse(workspace.isAssessingLearning)
        }
        lesson.submit();try await wait()
        XCTAssertEqual(lesson.grade?.status,.failed,lesson.statusMessage)
        XCTAssertNotNil(lesson.formativeProjection,"A valid wrong calculation retains measured feedback")
        XCTAssertTrue(lesson.drafts.update {$0.graph.nodes[1].config=task.configuration})
        XCTAssertFalse(lesson.resultIsCurrent,"Editing invalidates the previous calculation")
        lesson.submit();try await wait()
        XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
        XCTAssertTrue(lesson.resultSaved)
    }

    func testAllFourStrictEnvelopesCanonicalDraftsAndUnknownAnswerRefusals()throws {
        for task in AdaptivePoolingLearningTask.allCases {
            let graph=GraphDocument(name:"My regions",nodes:[.init(id:"x",type:"input",name:"Measurements",x:0,y:0),.init(id:"p",type:task.componentID,name:"Regions",x:200,y:0,config:task.configuration),.init(id:"y",type:"output",name:"Results",x:400,y:0)],edges:[.init(id:"first",source:"x",target:"p"),.init(id:"second",source:"p",target:"y")])
            let bytes=try AdaptivePoolingLearningSubmission(graph:graph,task:task).encoded()
            XCTAssertEqual(try AdaptivePoolingLearningSubmission.decode(bytes,task:task).encoded(),bytes)
            var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided);draft.graph=graph
            XCTAssertEqual(try draft.submissionData(),bytes)
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            XCTAssertThrowsError(try draft.submission(),"The generic envelope must not replace the reviewed one")
            for mutation in ["answer","kind","schema","configuration"] {
                var object=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
                if mutation == "answer" {object["expected"]=["output":[0]]}
                else if mutation == "kind" {object["kind"]="poolingGraph"}
                else if mutation == "schema" {object["schemaVersion"]=true}
                else {
                    var rawGraph=try XCTUnwrap(object["graph"] as? [String:Any])
                    var nodes=try XCTUnwrap(rawGraph["nodes"] as? [[String:Any]])
                    var config=try XCTUnwrap(nodes[1]["config"] as? [String:Any]);config["inplace"]=true
                    nodes[1]["config"]=config;rawGraph["nodes"]=nodes;object["graph"]=rawGraph
                }
                XCTAssertThrowsError(try AdaptivePoolingLearningSubmission.decode(JSONSerialization.data(withJSONObject:object),task:task),task.rawValue+" "+mutation)
            }
            draft.pendingFields=["p.unapplied":"3"]
            XCTAssertThrowsError(try draft.submissionData())
            draft.pendingFields=[:];draft.parameters=["p":["weight":.init(shape:[1],values:[1])]]
            XCTAssertThrowsError(try draft.validate(),"These pooling lessons do not own learned parameters")
        }
    }

}
