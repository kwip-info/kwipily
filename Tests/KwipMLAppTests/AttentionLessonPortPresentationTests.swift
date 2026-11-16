import XCTest
import CryptoKit
import KwipMLCore
@testable import KwipMLApp
final class AttentionLessonPortPresentationTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    @MainActor func testUnwiredAttentionPortsDeclareRequiredShapesAndWholeNumberMaskRoles() throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        for task in AttentionLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue));let canvas=try XCTUnwrap(lesson.canvas)
            canvas.addNode(type:task.componentID,x:0,y:0);canvas.addNode(type:"attention_mask",x:100,y:0)
            let op=try XCTUnwrap(canvas.graph.nodes.first {$0.type == task.componentID}),mask=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "attention_mask"})
            for role in task.sourceRoles {
                let d=canvas.canvasPortDescriptor(nodeID:op.id,port:role,direction:.input)
                XCTAssertTrue(d.declared.contains("float64"));XCTAssertTrue(d.declared.contains { $0.contains("Required lesson shape") && $0.contains(try! BatchedTensorLayout(sample:task.sampleShapes[role]!).displayShape) });XCTAssertFalse(d.accessibilityDescription.contains("unresolved"));XCTAssertTrue(d.measured.isEmpty)
            }
            let d=canvas.canvasPortDescriptor(nodeID:op.id,port:task.maskInputPort,direction:.input)
            XCTAssertTrue(d.declared.contains("int64"));XCTAssertEqual(d.role,"Blocked connections");XCTAssertTrue(d.declared.contains {$0.contains("1 blocks")});XCTAssertTrue(d.declared.contains {$0.contains("Required lesson shape")});XCTAssertFalse(d.accessibilityDescription.contains("floating number source"));XCTAssertTrue(d.measured.isEmpty)
            for (port,role) in [("mask","Blocked connections"),("padding","Blocked source positions")] {
                let d=canvas.canvasPortDescriptor(nodeID:mask.id,port:port,direction:.output)
                XCTAssertEqual(d.role,role);XCTAssertTrue(d.declared.contains("int64"));XCTAssertTrue(d.declared.contains {$0.contains("Required lesson shape")});XCTAssertTrue(d.measured.isEmpty)
            }
            for role in task.outputRoles {let d=canvas.canvasPortDescriptor(nodeID:op.id,port:role == "result" ? "out":role,direction:.output);XCTAssertTrue(d.declared.contains("float64"));XCTAssertTrue(d.declared.contains {$0.contains("Required lesson shape")})}
        }
    }
    @MainActor func testDisabledMaskAndWeightPortsStayUnknown() throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:AttentionLearningTask.multihead.rawValue));let canvas=try XCTUnwrap(lesson.canvas)
        canvas.addNode(type:"multihead_attention",x:0,y:0);let op=try XCTUnwrap(canvas.graph.nodes.first)
        var config=op.config;config["attention_mask_mode"] = .string("none");config["padding_mask_mode"] = .string("none");config["return_weights"] = .bool(false)
        XCTAssertTrue(canvas.configure(nodeID:op.id,values:config))
        for (port,direction) in [("attn_mask",PortDirection.input),("key_padding_mask",.input),("weights",.output)] {XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:op.id,port:port,direction:direction).kind,.unknown)}
    }
    @MainActor func testWiredUnresolvedMaskDoesNotPretendRequirementWasInferred() throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:AttentionLearningTask.multihead.rawValue));let canvas=try XCTUnwrap(lesson.canvas)
        canvas.addNode(type:"multihead_attention",x:0,y:0);canvas.addNode(type:"attention_mask",x:100,y:0)
        let op=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "multihead_attention"}),mask=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "attention_mask"})
        canvas.connect(source:mask.id,sourcePort:"mask",target:op.id,targetPort:"attn_mask")
        XCTAssertEqual(canvas.graph.edges.count,1)
        let d=canvas.canvasPortDescriptor(nodeID:op.id,port:"attn_mask",direction:.input)
        XCTAssertTrue(d.declared.contains("Shape unresolved"));XCTAssertFalse(d.declared.contains {$0.contains("Required lesson shape")});XCTAssertEqual(d.role,"Blocked connections");XCTAssertFalse(d.accessibilityDescription.contains("floating number source"));XCTAssertTrue(d.measured.isEmpty)
    }
}

extension AttentionLessonPortPresentationTests {
    @MainActor func testValidatedCurrentAttentionTablesMapToExactPortsAndDenyPrivateStaleMissing() throws {
        // Presentation test over a structurally verified receipt, not native
        // numerical execution evidence (covered by the actual lesson lifecycle).
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace),task=AttentionLearningTask.multihead
        XCTAssertTrue(lesson.start(operation:task.rawValue))
        var nodes=task.sourceRoles.map {GraphNode(id:$0,type:"input",name:$0,config:["fixture":.string($0)])}
        nodes += [.init(id:"mask",type:"attention_mask",name:"Mask",config:task.maskConfiguration),.init(id:"op",type:task.componentID,name:"Attention",config:task.configuration)]
        nodes += task.outputRoles.map {GraphNode(id:"output-"+$0,type:"output",name:$0,config:["role":.string($0)])}
        var edges=task.sourceRoles.map {GraphEdge(id:$0,source:$0,sourcePort:"out",target:"op",targetPort:$0)}
        edges += [.init(id:"query-mask",source:"query",sourcePort:"out",target:"mask",targetPort:"query"),.init(id:"key-mask",source:"key",sourcePort:"out",target:"mask",targetPort:"key"),.init(id:"mask-op",source:"mask",sourcePort:"mask",target:"op",targetPort:"attn_mask"),.init(id:"pad-op",source:"mask",sourcePort:"padding",target:"op",targetPort:"key_padding_mask")]
        edges += task.outputRoles.map {GraphEdge(id:"output-"+$0,source:"op",sourcePort:$0 == "result" ? "out":$0,target:"output-"+$0,targetPort:"in")}
        XCTAssertTrue(lesson.drafts.update {$0.graph = .init(name:"Presentation",nodes:nodes,edges:edges)})
        let canvas=try XCTUnwrap(lesson.canvas),draft=try XCTUnwrap(lesson.draft),data=try draft.submissionData(),plan=try AttentionAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:41)
        let digest=try AttentionLearningSubmission.decode(data,task:task).encoded()
        // Use actual receipt constructor and verified projection boundary.
        let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:SHA256.hash(data:digest).map {String(format:"%02x",$0)}.joined(),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:36)
        let receipt=try TensorLearningReceipt(projectID:draft.projectID,operation:task.rawValue,mode:.guided,submissionData:data,plan:plan,evidence:evidence)
        let projection=try XCTUnwrap(receipt.attentionFormativeProjection(expectedProjectID:draft.projectID,currentSubmission:data,currentMode:.guided))
        canvas.attentionProjectionProvider={projection};canvas.evidenceState={.current}
        XCTAssertEqual(canvas.previewStatus(for:"mask"),"Measured mask/positions");XCTAssertEqual(canvas.previewStatus(for:"op"),"Measured results")
        for (node,port) in [("mask","mask"),("mask","padding"),("op","out"),("op","weights")] {XCTAssertFalse(canvas.canvasPortDescriptor(nodeID:node,port:port,direction:.output).measured.isEmpty)}
        XCTAssertTrue(canvas.canvasPortDescriptor(nodeID:"op",port:"phantom",direction:.output).measured.isEmpty)
        XCTAssertFalse(canvas.canvasPortDescriptor(nodeID:"op",port:"attn_mask",direction:.input).measured.isEmpty)
        XCTAssertFalse(canvas.canvasPortDescriptor(nodeID:"op",port:"key_padding_mask",direction:.input).measured.isEmpty)
        canvas.evidenceState={.stale};XCTAssertEqual(canvas.previewStatus(for:"mask"),"Stale");XCTAssertTrue(canvas.canvasPortDescriptor(nodeID:"mask",port:"mask",direction:.output).measured.isEmpty)
        canvas.evidenceState={.current};XCTAssertTrue(lesson.setMode(.assessment));XCTAssertTrue(canvas.canvasPortDescriptor(nodeID:"op",port:"out",direction:.output).measured.isEmpty);XCTAssertFalse(canvas.previewStatus(for:"mask").hasPrefix("Measured"))
        XCTAssertTrue(lesson.setMode(.guided));var old=receipt;old.evidence.fixtures[0].tensors.removeValue(forKey:"blockedConnections")
        XCTAssertNil(old.attentionFormativeProjection(expectedProjectID:draft.projectID,currentSubmission:data,currentMode:.guided))
        canvas.attentionProjectionProvider={nil};XCTAssertTrue(canvas.canvasPortDescriptor(nodeID:"mask",port:"padding",direction:.output).measured.isEmpty);XCTAssertFalse(canvas.previewStatus(for:"mask").hasPrefix("Measured"))
    }
}
