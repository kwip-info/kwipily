import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TensorCompositionLearningStoreTests:XCTestCase {
    @MainActor func testAllFourActualGuidedPracticePrivateWorkersAndSavedReplay()async throws {
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
        for task in TensorCompositionLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue),lesson.statusMessage)
            let graph=composedGraph(task:task)
            XCTAssertTrue(lesson.drafts.update {$0.graph=graph})
            XCTAssertTrue(lesson.canSubmit,lesson.statusMessage)
            lesson.submit();try await wait()
            XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
            XCTAssertTrue(lesson.resultSaved);XCTAssertNotNil(lesson.formativeProjection)
            let id=try XCTUnwrap(workspace.activeProjectID),saved=workspace.graph
            XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Other work")))
            XCTAssertTrue(workspace.beginProject(saved,id:id))
            XCTAssertEqual(lesson.grade?.status,.passed);XCTAssertTrue(lesson.resultIsCurrent)
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
        }
    }
    @MainActor func testValidWrongPatternMultiplierIsMeasuredFailedThenRepaired()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        await verifyRuntimeForIntegration(workspace)
        let task=TensorCompositionLearningTask.add
        XCTAssertTrue(lesson.start(operation:task.rawValue),lesson.statusMessage)
        var wrong=task.configuration;wrong["alpha"]=1
        let graph=composedGraph(task:task,configuration:wrong)
        XCTAssertTrue(lesson.drafts.update {$0.graph=graph})
        XCTAssertTrue(lesson.canSubmit,"A valid pattern multiplier must be measured, not rejected as a malformed graph")
        func wait()async throws {
            let deadline=Date().addingTimeInterval(30)
            while workspace.isAssessingLearning,Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
            XCTAssertFalse(workspace.isAssessingLearning)
        }
        lesson.submit();try await wait()
        XCTAssertEqual(lesson.grade?.status,.failed,lesson.statusMessage)
        XCTAssertNotNil(lesson.formativeProjection,"A valid wrong calculation retains measured feedback")
        XCTAssertTrue(lesson.drafts.update {draft in
            let index=draft.graph.nodes.firstIndex(where:{$0.id == "op"})!
            draft.graph.nodes[index].config=task.configuration
        })
        XCTAssertFalse(lesson.resultIsCurrent,"Editing invalidates the previous calculation")
        lesson.submit();try await wait()
        XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
        XCTAssertTrue(lesson.resultSaved)
    }

    func testAllFourStrictEnvelopesCanonicalDraftsAndUnknownAnswerRefusals()throws {
        for task in TensorCompositionLearningTask.allCases {
            let graph=composedGraph(task:task)
            let bytes=try TensorCompositionLearningSubmission(graph:graph,task:task).encoded()
            XCTAssertEqual(try TensorCompositionLearningSubmission.decode(bytes,task:task).encoded(),bytes)
            var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided);draft.graph=graph
            XCTAssertEqual(try draft.submissionData(),bytes)
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            XCTAssertThrowsError(try draft.submission(),"The generic envelope must not replace the reviewed one")
            for mutation in ["answer","kind","schema","configuration"] {
                var object=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
                if mutation == "answer" {object["expected"]=["output":[0]]}
                else if mutation == "kind" {object["kind"]="adaptivePoolingGraph"}
                else if mutation == "schema" {object["schemaVersion"]=true}
                else {
                    var rawGraph=try XCTUnwrap(object["graph"] as? [String:Any])
                    var nodes=try XCTUnwrap(rawGraph["nodes"] as? [[String:Any]])
                    var config=try XCTUnwrap(nodes.first(where:{$0["id"] as? String == "op"})!["config"] as? [String:Any]);config["inplace"]=true
                    nodes[nodes.firstIndex(where:{$0["id"] as? String == "op"})!]["config"]=config;rawGraph["nodes"]=nodes;object["graph"]=rawGraph
                }
                XCTAssertThrowsError(try TensorCompositionLearningSubmission.decode(JSONSerialization.data(withJSONObject:object),task:task),task.rawValue+" "+mutation)
            }
            draft.pendingFields=["p.unapplied":"3"]
            XCTAssertThrowsError(try draft.submissionData())
            draft.pendingFields=[:];draft.parameters=["p":["weight":.init(shape:[1],values:[1])]]
            XCTAssertThrowsError(try draft.validate(),"These composition lessons do not own learned parameters")
        }
    }

    private func composedGraph(task:TensorCompositionLearningTask,configuration:[String:JSONValue]?=nil)->GraphDocument {
        var nodes=[GraphNode(id:"left",type:"input",name:"First source",x:0,y:0,config:["fixture":.string("left")])]
        var edges=[GraphEdge(id:"left-wire",source:"left",target:"op",targetPort:task.isBinary ? "a":"in")]
        if task.isBinary {
            nodes.append(.init(id:"right",type:"input",name:"Second source",x:0,y:180,config:["fixture":.string("right")]))
            edges.append(.init(id:"right-wire",source:"right",target:"op",targetPort:"b"))
        }
        nodes.append(.init(id:"op",type:task.componentID,name:"My arrangement",x:200,y:0,config:configuration ?? task.configuration))
        nodes.append(.init(id:"out",type:"output",name:"Arranged values",x:400,y:0))
        edges.append(.init(id:"result-wire",source:"op",target:"out"))
        return .init(name:"Combine my sources",nodes:nodes,edges:edges)
    }

    @MainActor func testPaletteCreatesOrderedSourcesAndInfersEveryTargetWithoutAnswerChecking()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(storageOverride:storage,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        for task in TensorCompositionLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue))
            let canvas=try XCTUnwrap(lesson.canvas)
            canvas.addNode(type:"input",x:0,y:0)
            if task.isBinary {canvas.addNode(type:"input",x:0,y:180)}
            XCTAssertFalse(canvas.canAddType("input"))
            XCTAssertEqual(Set(canvas.graph.nodes.compactMap {$0.config["fixture"]?.stringValue}),task.isBinary ? ["left","right"]:["left"])
            XCTAssertTrue(canvas.registry.components.contains {$0.id == task.componentID})
            XCTAssertFalse(canvas.registry.components.contains {$0.id == (task == .stack ? "add":"stack")})
            let graph=composedGraph(task:task)
            XCTAssertTrue(lesson.drafts.update {$0.graph=graph})
            XCTAssertEqual(canvas.validation.tensorLayouts["out"]?.sample.dimensions,task.outputSampleShape)
            XCTAssertTrue(canvas.validation.issues.isEmpty,canvas.validation.issues.map(\.message).joined(separator:"; "))
        }
    }

}
