import XCTest
import KwipMLCore
@testable import KwipMLApp
final class AttentionLearningIntegrationTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func graph(_ task: AttentionLearningTask) -> GraphDocument {
        var nodes = task.sourceRoles.map { GraphNode(id:$0,type:"input",name:$0,config:["fixture":.string($0)]) }
        nodes += [.init(id:"mask",type:"attention_mask",name:"Blocked connections",config:task.maskConfiguration),.init(id:"op",type:task.componentID,name:task.componentTitle,config:task.configuration)]
        nodes += task.outputRoles.map { .init(id:"output-"+$0,type:"output",name:$0,config:["role":.string($0)]) }
        var edges = task.sourceRoles.map { GraphEdge(id:"in-"+$0,source:$0,target:"op",targetPort:$0) }
        for (i,role) in task.maskReferenceRoles.enumerated() { edges.append(.init(id:"ref-\(i)",source:role,target:"mask",targetPort:i == 0 ? "query":"key")) }
        edges.append(.init(id:"mask",source:"mask",sourcePort:"mask",target:"op",targetPort:task.maskInputPort))
        if task == .multihead { edges.append(.init(id:"padding",source:"mask",sourcePort:"padding",target:"op",targetPort:"key_padding_mask")) }
        edges += task.outputRoles.map { .init(id:"out-"+$0,source:"op",sourcePort:$0 == "result" ? "out":$0,target:"output-"+$0) }
        return .init(schemaVersion:2,name:"Compare useful values",nodes:nodes,edges:edges)
    }
    @MainActor private func wait(_ workspace: WorkspaceStore) async throws { let deadline = Date().addingTimeInterval(45); while workspace.isAssessingLearning, Date()<deadline { try await Task.sleep(for:.milliseconds(20)) }; XCTAssertFalse(workspace.isAssessingLearning) }
    @MainActor func testEveryAttentionTaskWorkerColdReceiptAndRepresentativeThreeModeLifecycle() async throws {
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString); defer { try? FileManager.default.removeItem(at:storage) }
        let workspace = WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false), lesson = TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:AttentionLearningTask.multihead.rawValue)); XCTAssertNotNil(lesson.runtimeReadinessError)
        await verifyRuntimeForIntegration(workspace)
        for task in AttentionLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue)); XCTAssertNil(lesson.runtimeReadinessError)
            XCTAssertTrue(lesson.drafts.update { $0.graph = self.graph(task) })
            // The shared runner plan has no mode-specific numerical dispatch.
            // Multihead retains all actual mode paths; every task retains its worker.
            let workerModes:[LearningMode] = task == .multihead ? [.guided,.practice,.assessment]:[.guided]
            for mode in workerModes {
                XCTAssertTrue(lesson.setMode(mode)); XCTAssertTrue(lesson.canSubmit,lesson.statusMessage); lesson.submit(); try await wait(workspace)
                if lesson.grade?.status != .passed,let diagnosticPath=ProcessInfo.processInfo.environment["KWIP_ATTENTION_LESSON_DIAGNOSTICS"],let projectID=lesson.draft?.projectID {
                    let repository=TensorLearningReceiptRepository(directory:workspace.workspaceDirectory.appendingPathComponent("learning/projects"))
                    if let receipt=try repository.latest(projectID:projectID) {
                        let directory=URL(fileURLWithPath:diagnosticPath)
                        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
                        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
                        try encoder.encode(receipt).write(to:directory.appendingPathComponent(task.rawValue+"-"+mode.rawValue+"-"+UUID().uuidString+".json"),options:.atomic)
                    }
                }
                XCTAssertEqual(lesson.grade?.status,.passed,"\(task.rawValue):\(lesson.statusMessage)"); XCTAssertTrue(lesson.resultSaved)
                XCTAssertEqual(lesson.attentionProjection == nil,mode == .assessment)
                if let projection = lesson.attentionProjection { XCTAssertEqual(projection.fixtures.count,4); for f in projection.fixtures { for role in task.outputRoles { XCTAssertNotNil(f.measured[role]) }; for key in task.sourceRoles+task.parameterShapes.keys.map({"parameter."+$0}) { XCTAssertEqual(f.measured[key+"Gradient"]?.shape,f.inputs[key]?.shape) }; XCTAssertFalse(f.measured.keys.contains { $0.hasPrefix("mask") }); XCTAssertNotNil(f.measured["blockedConnections"]); XCTAssertNotNil(f.measured["blockedSourcePositions"]) } }
            }
            let id = try XCTUnwrap(workspace.activeProjectID), document = workspace.graph, cold = WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false), restored = TensorLearningStore(workspace:cold)
            await verifyRuntimeForIntegration(cold); XCTAssertTrue(cold.beginProject(document,id:id)); XCTAssertEqual(restored.grade?.status,.passed); XCTAssertTrue(restored.resultSaved); XCTAssertEqual(restored.attentionProjection == nil,task == .multihead)
            let saved=TensorLearningDraftRepository(directory:storage.appendingPathComponent("learning/projects"))
            for mode in [LearningMode.practice,.assessment,.guided] {
                XCTAssertTrue(lesson.setMode(mode));XCTAssertTrue(lesson.save())
                XCTAssertNil(lesson.grade);XCTAssertNil(lesson.attentionProjection);XCTAssertFalse(lesson.resultSaved)
                XCTAssertEqual(try saved.load(projectID:id).draft.mode,mode)
                if task == .multihead {
                    let reopened=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
                    let reopenedLesson=TensorLearningStore(workspace:reopened)
                    XCTAssertTrue(reopened.beginProject(document,id:id))
                    XCTAssertEqual(reopenedLesson.draft?.mode,mode)
                    if mode == .assessment {XCTAssertNil(reopenedLesson.attentionProjection)}
                }
            }
        }
    }
    @MainActor func testWrongMaskIsMeasuredFailureRepairAndStaleProjectionRefusal() async throws {
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString); defer { try? FileManager.default.removeItem(at:storage) }
        let workspace = WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false), lesson = TensorLearningStore(workspace:workspace); await verifyRuntimeForIntegration(workspace)
        for task in AttentionLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue)); var wrong = graph(task); wrong.nodes[task.sourceRoles.count].config["blocked_queries"] = .array([1])
            XCTAssertTrue(lesson.drafts.update { $0.graph = wrong }); XCTAssertTrue(lesson.canSubmit); lesson.submit(); try await wait(workspace)
            if lesson.grade?.status != .failed,let path=ProcessInfo.processInfo.environment["KWIP_ATTENTION_LESSON_DIAGNOSTICS"],let projectID=lesson.draft?.projectID {
                let repository=TensorLearningReceiptRepository(directory:workspace.workspaceDirectory.appendingPathComponent("learning/projects"))
                if let receipt=try repository.latest(projectID:projectID) {
                    let directory=URL(fileURLWithPath:path);try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
                    let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
                    try encoder.encode(receipt).write(to:directory.appendingPathComponent("wrong-mask-"+task.rawValue+"-"+UUID().uuidString+".json"),options:.atomic)
                }
            }
            XCTAssertEqual(lesson.grade?.status,.failed,task.rawValue+": "+lesson.statusMessage); XCTAssertTrue(lesson.resultSaved)
            XCTAssertTrue(lesson.grade?.outcomes.contains {$0.criterionID == "blockedConnections" && !$0.passed} == true)
            XCTAssertTrue(lesson.grade?.outcomes.contains {$0.criterionID == "blockedSourcePositions" && $0.passed} == true)
            XCTAssertTrue(lesson.drafts.update { $0.graph = self.graph(task) }); XCTAssertFalse(lesson.resultIsCurrent); XCTAssertNil(lesson.attentionProjection); lesson.submit(); try await wait(workspace); XCTAssertEqual(lesson.grade?.status,.passed)
            var cheat = graph(task); cheat.nodes[0].config["answer"] = 1; XCTAssertFalse(lesson.drafts.update { $0.graph = cheat })
        }
    }
    @MainActor func testDedicatedPaletteRoleAssignmentMaskPortsAndPendingControls() throws {
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString); defer { try? FileManager.default.removeItem(at:storage) }
        let workspace = WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false), lesson = TensorLearningStore(workspace:workspace)
        for task in AttentionLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue)); let canvas = try XCTUnwrap(lesson.canvas)
            XCTAssertEqual(Set(canvas.registry.components.map(\.id)),["input","output",task.componentID,"attention_mask"])
            for _ in task.sourceRoles { canvas.addNode(type:"input",x:0,y:0) }; canvas.addNode(type:"attention_mask",x:80,y:0); canvas.addNode(type:task.componentID,x:160,y:0); for _ in task.outputRoles { canvas.addNode(type:"output",x:240,y:0) }
            XCTAssertEqual(Set(canvas.graph.nodes.filter { $0.type == "input" }.compactMap { $0.config["fixture"]?.stringValue }),Set(task.sourceRoles))
            XCTAssertEqual(Set(canvas.graph.nodes.filter { $0.type == "output" }.compactMap { $0.config["role"]?.stringValue }),Set(task.outputRoles))
            let op = try XCTUnwrap(canvas.graph.nodes.first { $0.type == task.componentID }), mask = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "attention_mask" })
            func ownedID(_ fixtureID: String) throws -> String {
                if fixtureID == "op" { return op.id }; if fixtureID == "mask" { return mask.id }
                let output = fixtureID.hasPrefix("output-"), role = output ? String(fixtureID.dropFirst(7)) : fixtureID
                return try XCTUnwrap(canvas.graph.nodes.first { $0.type == (output ? "output" : "input") && $0.config[output ? "role" : "fixture"]?.stringValue == role }).id
            }
            for edge in graph(task).edges { canvas.connect(source:try ownedID(edge.source),sourcePort:edge.sourcePort,target:try ownedID(edge.target),targetPort:edge.targetPort) }
            XCTAssertTrue(canvas.validation.isValid,canvas.statusMessage)
            for role in task.outputRoles { XCTAssertEqual(canvas.validation.outputTensorContracts[.init(op.id,role == "result" ? "out":role)]?.layout.sample.dimensions,task.outputSampleShapes[role]) }
            XCTAssertEqual(canvas.validation.outputTensorContracts[.init(mask.id,"mask")]?.dtype,.int64)
            XCTAssertEqual(op.config,try AttentionConfiguration(componentID:task.componentID,values:task.configuration).values)
            let edgeCount = canvas.graph.edges.count; canvas.undo(); XCTAssertEqual(canvas.graph.edges.count,edgeCount-1); canvas.redo(); XCTAssertEqual(canvas.graph.edges.count,edgeCount)
            XCTAssertTrue(canvas.setPending(nodeID:mask.id,field:"blocked_keys",text:"unfinished")); XCTAssertThrowsError(try lesson.draft?.submissionData()); XCTAssertFalse(canvas.applyPending(nodeID:mask.id,field:"blocked_keys")); XCTAssertEqual(canvas.pendingText(nodeID:mask.id,field:"blocked_keys"),"unfinished"); canvas.discardPending(nodeID:mask.id,field:"blocked_keys")
            XCTAssertNoThrow(try lesson.draft?.submissionData())
        }
    }
}

extension AttentionLearningIntegrationTests {
    @MainActor func testFiveSearchableExamplesAndEveryCriterionHasApproachableGuidance() throws {
        let ids = Set(AttentionLearningTask.allCases.map(\.componentID))
        let entries = LearnCatalogContent.displayedComponentExampleEntries.filter { ids.contains($0.componentID) }
        XCTAssertEqual(entries.count,5); XCTAssertEqual(Set(entries.map(\.componentID)),ids)
        for task in AttentionLearningTask.allCases {
            let entry = try XCTUnwrap(entries.first { $0.componentID == task.componentID })
            XCTAssertEqual(entry.item.kind,"Example"); XCTAssertTrue(entry.item.matches(section:"Examples",search:task.componentTitle))
            let guide = LessonAudienceGuide.make(operation:task.rawValue)
            XCTAssertEqual(guide.goal,task.title); XCTAssertEqual(guide.steps.count,3); XCTAssertTrue(guide.steps[0].contains("mask")); XCTAssertTrue(guide.steps[1].contains("Heads") || guide.steps[1].contains("Automatic scale"))
            let requirements = LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"")
            for heading in ["**Build**","**Configure**","**Check**"] { XCTAssertTrue(requirements.contains(heading)) }
            for term in ["Query","Key","Value","Mask","Head"] { XCTAssertTrue(guide.terms.contains { $0.name == term }) }
            let plan = try AttentionAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:41)
            for criterion in plan.criteria { XCTAssertNotNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:criterion.id)) }
        }
    }
}

extension AttentionLearningIntegrationTests {
    @MainActor func testEmptyMaskListCanBeAppliedAndDiscardedWithoutLosingPendingText() throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:AttentionLearningTask.scaled.rawValue))
        let canvas=try XCTUnwrap(lesson.canvas)
        canvas.addNode(type:"attention_mask",x:0,y:0)
        let mask=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "attention_mask"})
        var configured=mask.config;configured["padding_keys"] = .array([2])
        XCTAssertTrue(canvas.configure(nodeID:mask.id,values:configured))
        XCTAssertTrue(canvas.setPending(nodeID:mask.id,field:"padding_keys",text:""))
        XCTAssertTrue(canvas.applyPending(nodeID:mask.id,field:"padding_keys"))
        XCTAssertEqual(canvas.graph.nodes.first {$0.id == mask.id}?.config["padding_keys"],.array([]))
        XCTAssertNil(canvas.pendingText(nodeID:mask.id,field:"padding_keys"))
        XCTAssertTrue(canvas.setPending(nodeID:mask.id,field:"padding_keys",text:"1, unfinished"))
        XCTAssertFalse(canvas.applyPending(nodeID:mask.id,field:"padding_keys"))
        XCTAssertEqual(canvas.pendingText(nodeID:mask.id,field:"padding_keys"),"1, unfinished")
        XCTAssertEqual(canvas.graph.nodes.first {$0.id == mask.id}?.config["padding_keys"],.array([]))
        canvas.discardPending(nodeID:mask.id,field:"padding_keys")
        XCTAssertNil(canvas.pendingText(nodeID:mask.id,field:"padding_keys"))
        XCTAssertEqual(canvas.graph.nodes.first {$0.id == mask.id}?.config["padding_keys"],.array([]))
    }
}
