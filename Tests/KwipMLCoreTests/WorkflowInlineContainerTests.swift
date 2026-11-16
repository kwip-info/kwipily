import XCTest
@testable import KwipMLCore

final class WorkflowInlineContainerTests:XCTestCase {
    private let root=WorkflowNodeAddress(domain:.model,localID:"features")
    private func fixture()throws->WorkflowDocument {
        let source=CompositeExamples.parallelClassifier
        let model=ModelAuthoringDocument(projectID:UUID().uuidString,revision:4,source:source)
        var document=try WorkflowCodec.migrateModel(.blockModel(model),projectID:model.projectID).document
        document.schemaVersion=11
        document.definitions[0].composition = .init(kind:.graph,children:document.definitions[0].body.nodes.map {.init(localID:$0.id)})
        return document
    }
    func testProxyUsesActualResolvedChildAndCanonicalIdentity()throws {
        var document=try fixture()
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==root})
        guard case .modelBlock(let pin,_)=document.nodes[index].payload else {return XCTFail("Missing block")}
        document.nodes[index].payload = .modelBlock(reference:pin,overrides:["branchWidth":12])
        let proxy=try WorkflowInlineContainer.node(in:document,root:root,childID:"branch-a",registry:.builtIn())
        guard case .modelPrimitive(let type,_,let config)=proxy.payload else {return XCTFail("Expected actual primitive")}
        XCTAssertEqual(type,"linear");XCTAssertEqual(config["out_features"],12)
        let address=WorkflowInlineContainer.address(root:root,childID:"branch-a")
        XCTAssertEqual(proxy.address,address)
        XCTAssertEqual(try WorkflowInlineContainer.parse(in:document,address:address)?.root,root)
        XCTAssertEqual(try WorkflowInlineContainer.parse(in:document,address:address)?.childID,"branch-a")
        XCTAssertNil(try WorkflowInlineContainer.parse(in:document,address:root))
    }
    func testRenameCopiesOnlySelectedDefinitionAndPreservesOriginalSource()throws {
        var document=try fixture()
        let selected=try XCTUnwrap(document.nodes.first {$0.address==root})
        document.nodes.append(.init(address:.init(domain:.model,localID:"other"),payload:selected.payload))
        let original=try WorkflowCodec.encode(document),oldDefinitions=document.definitions
        let edited=try WorkflowInlineContainer.apply(.rename(childID:"activation-a",name:"My activation"),to:document,root:root,registry:.builtIn(),idProvider:{"private-copy"})
        XCTAssertEqual(edited.revision,document.revision+1)
        XCTAssertEqual(Array(edited.definitions.prefix(oldDefinitions.count)),oldDefinitions)
        XCTAssertEqual(edited.nodes.first {$0.address.localID=="other"}?.payload,selected.payload)
        XCTAssertEqual(try WorkflowInlineContainer.node(in:edited,root:root,childID:"activation-a",registry:.builtIn()).name,"My activation")
        XCTAssertEqual(try WorkflowCodec.encode(document),original)
        XCTAssertEqual(edited.schemaVersion,document.schemaVersion)
    }
    func testBoundLinearMoveOutFreezesEffectiveValuesAndRetainsSharedSiblingControl()throws {
        var document=try fixture()
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==root})
        guard case .modelBlock(let pin,_)=document.nodes[index].payload else {return XCTFail("Missing block")}
        document.nodes[index].payload = .modelBlock(reference:pin,overrides:["branchWidth":12])
        let moved=WorkflowNodeAddress(domain:.model,localID:"movedLinear"),original=try WorkflowCodec.encode(document)
        var serial=0
        let result=try WorkflowInlineContainer.apply(.moveOut(childID:"branch-a",address:moved),to:document,root:root,registry:.builtIn(),idProvider:{serial+=1;return "transfer-\(serial)"})
        guard case .modelPrimitive(let type,_,let config)=try XCTUnwrap(result.nodes.first {$0.address==moved}).payload else {return XCTFail("Missing moved primitive")}
        XCTAssertEqual(type,"linear");XCTAssertEqual(config["out_features"],12)
        let definition=try WorkflowInlineContainer.definition(in:result,root:root)
        XCTAssertEqual(definition.parameters.first {$0.field.key=="branchWidth"}?.targets,[.init("branch-b","out_features")])
        XCTAssertTrue(result.edges.contains {$0.from.address==moved && $0.to.address==root})
        XCTAssertTrue(result.edges.contains {$0.from.address.localID=="input" && $0.to.address==moved})
        XCTAssertEqual(try WorkflowInlineContainer.node(in:result,root:root,childID:"branch-b",registry:.builtIn()).payload,try WorkflowInlineContainer.node(in:document,root:root,childID:"branch-b",registry:.builtIn()).payload)
        XCTAssertEqual(try WorkflowCodec.encode(document),original)
    }
    func testDisconnectedMembershipAndParentChildWiringStayEditableAndAtomic()throws {
        var document=try fixture()
        let member=WorkflowNodeAddress(domain:.model,localID:"newActivation")
        document.nodes.append(.init(address:member,payload:.modelPrimitive(type:"relu",configurationVersion:2,config:[:])))
        var serial=0
        let moved=try WorkflowInlineContainer.apply(.moveIn(address:member,position:.init(x:80,y:120)),to:document,root:root,registry:.builtIn(),idProvider:{serial+=1;return "membership-\(serial)"})
        XCTAssertFalse(moved.nodes.contains {$0.address==member})
        let proxy=try WorkflowInlineContainer.node(in:moved,root:root,childID:member.localID,registry:.builtIn())
        XCTAssertEqual(proxy.x,80);XCTAssertEqual(proxy.y,120)
        let wired=try WorkflowInlineContainer.apply(.connectWorkflow(from:.init(address:.init(domain:.model,localID:"input"),port:"out"),to:.init(address:proxy.address,port:"in")),to:moved,root:root,registry:.builtIn(),idProvider:{serial+=1;return "membership-\(serial)"})
        let definition=try WorkflowInlineContainer.definition(in:wired,root:root)
        let exposed=try XCTUnwrap(definition.inputs.first {$0.targets.contains(.init(member.localID,"in"))})
        XCTAssertTrue(wired.edges.contains {$0.from.address.localID=="input" && $0.to == .init(address:root,port:exposed.id)})
        let bytes=try WorkflowCodec.encode(wired)
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.connectWorkflow(from:.init(address:.init(domain:.model,localID:"input"),port:"out"),to:.init(address:proxy.address,port:"in")),to:wired,root:root,registry:.builtIn(),idProvider:{serial+=1;return "membership-\(serial)"}))
        XCTAssertEqual(try WorkflowCodec.encode(wired),bytes)
    }
    func testBoundConfigurationRefusalAndIncompleteDeletePreserveAuthority()throws {
        let document=try fixture(),proxy=try WorkflowInlineContainer.node(in:document,root:root,childID:"branch-a",registry:.builtIn())
        guard case .modelPrimitive(_,_,var config)=proxy.payload else {return XCTFail("Missing primitive")}
        config["out_features"]=16
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.configure(childID:"branch-a",config:config),to:document,root:root,registry:.builtIn(),idProvider:{"refused"}))
        let result=try WorkflowInlineContainer.apply(.delete(childID:"activation-a"),to:document,root:root,registry:.builtIn(),idProvider:{"deleted"})
        XCTAssertFalse(try WorkflowInlineContainer.definition(in:result,root:root).body.nodes.contains {$0.id=="activation-a"})
        XCTAssertNil(try WorkflowInlineContainer.parse(in:result,address:WorkflowInlineContainer.address(root:root,childID:"activation-a")))
        XCTAssertTrue(result.drafts.isEmpty,"Incomplete authoring remains editable without a hidden draft lock")
    }
    func testOneHundredSessionEditsRetireOnlyProvenPrivatePredecessors()throws {
        var current=try fixture()
        let owner=try XCTUnwrap(current.nodes.first {$0.address==root})
        current.nodes.append(.init(address:.init(domain:.model,localID:"sharedOriginal"),payload:owner.payload))
        let archived=try WorkflowCodec.encode(current),originalDefinitions=current.definitions
        var previous:CompositeDefinitionReference?,serial=0
        for ordinal in 0..<100 {
            let command:WorkflowInlineContainer.Command
            if ordinal.isMultiple(of:2) {
                let proxy=try WorkflowInlineContainer.node(in:current,root:root,childID:"branch-a",registry:.builtIn())
                guard case .modelPrimitive(_,_,var config)=proxy.payload else {return XCTFail("Missing primitive")}
                config["bias"] = .bool(ordinal % 4 == 2)
                command = .configure(childID:"branch-a",config:config)
            } else {command = .rename(childID:"activation-a",name:"Activation \(ordinal)")}
            current=try WorkflowInlineContainer.apply(command,to:current,root:root,registry:.builtIn(),retirePreviousPrivateReference:previous,idProvider:{serial+=1;return "retained-\(serial)"})
            previous=try WorkflowInlineContainer.definition(in:current,root:root).reference
            XCTAssertEqual(current.definitions.count,originalDefinitions.count+1)
            XCTAssertEqual(Array(current.definitions.prefix(originalDefinitions.count)),originalDefinitions)
        }
        XCTAssertEqual(current.nodes.first {$0.address.localID=="sharedOriginal"}?.payload,owner.payload)
        XCTAssertEqual(try WorkflowCodec.decode(archived).definitions,originalDefinitions)
        XCTAssertEqual(try WorkflowCodec.encode(WorkflowCodec.decode(archived)),archived)
        // A genuinely compiled primitive graph remains available after repeated edits.
        current.nodes.removeAll {$0.address.localID=="sharedOriginal"}
        let project=CompositeProjectDocument(schemaVersion:3,root:.init(nodes:current.nodes.compactMap {node in
            guard case .modelPrimitive(let type,_,let config)=node.payload else {return nil}
            return GraphNode(id:node.address.localID,type:type,config:config)
        },instances:current.nodes.compactMap {node in
            guard case .modelBlock(let reference,let overrides)=node.payload else {return nil}
            return CompositeInstance(id:node.address.localID,definition:reference,overrides:overrides)
        },edges:current.edges.map {GraphEdge(id:$0.id,source:$0.from.address.localID,sourcePort:$0.from.port,target:$0.to.address.localID,targetPort:$0.to.port)}),definitions:current.definitions)
        XCTAssertEqual(try CompositeExpander.expand(project,registry:.builtIn()).graph.nodes.count,8)
    }
    func testPrivatePredecessorSharedByAnotherOwnerIsNeverRetired()throws {
        let original=try fixture()
        var first=try WorkflowInlineContainer.apply(.rename(childID:"activation-a",name:"First"),to:original,root:root,registry:.builtIn(),idProvider:{"first-private"})
        let pin=try WorkflowInlineContainer.definition(in:first,root:root).reference
        let owner=try XCTUnwrap(first.nodes.first {$0.address==root})
        first.nodes.append(.init(address:.init(domain:.model,localID:"sharedPrivate"),payload:owner.payload))
        let next=try WorkflowInlineContainer.apply(.rename(childID:"activation-a",name:"Second"),to:first,root:root,registry:.builtIn(),retirePreviousPrivateReference:pin,idProvider:{"second-private"})
        XCTAssertTrue(next.definitions.contains {$0.reference==pin})
        XCTAssertEqual(next.nodes.first {$0.address.localID=="sharedPrivate"}?.payload,owner.payload)
    }
    func testLegacyParallelFeaturesProjectsWithoutMutationAndEditsOnlyPrivateCopy()throws {
        let model=ModelAuthoringDocument(projectID:UUID().uuidString,revision:4,source:CompositeExamples.parallelClassifier)
        let legacy=try WorkflowCodec.migrateModel(.blockModel(model),projectID:model.projectID).document
        let bytes=try WorkflowCodec.encode(legacy)
        XCTAssertNil(legacy.definitions[0].composition)
        let transient=try WorkflowInlineContainer.definition(in:legacy,root:root)
        XCTAssertEqual(transient.composition?.kind,.graph)
        let child=try WorkflowInlineContainer.node(in:legacy,root:root,childID:"branch-a",registry:.builtIn())
        XCTAssertEqual(try WorkflowInlineContainer.parse(in:legacy,address:child.address)?.childID,"branch-a")
        XCTAssertEqual(try WorkflowCodec.encode(legacy),bytes)
        let edited=try WorkflowInlineContainer.apply(.rename(childID:"branch-a",name:"My linear"),to:legacy,root:root,registry:.builtIn(),idProvider:{"legacy-edit"})
        XCTAssertGreaterThanOrEqual(edited.schemaVersion,11)
        XCTAssertEqual(edited.definitions[0],legacy.definitions[0])
        XCTAssertNil(edited.definitions[0].composition)
        XCTAssertEqual(try WorkflowInlineContainer.definition(in:edited,root:root).composition?.kind,.graph)
        XCTAssertEqual(try WorkflowInlineContainer.node(in:edited,root:root,childID:"branch-a",registry:.builtIn()).name,"My linear")
        XCTAssertEqual(try WorkflowCodec.encode(legacy),bytes)
    }
    func testInteriorMoveOutRefusesScopeCycleWithoutBlockingNextValidTransfer()throws {
        let original=try fixture(),bytes=try WorkflowCodec.encode(original)
        var serial=0
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.moveOut(childID:"activation-a",address:.init(domain:.model,localID:"outsideActivation")),to:original,root:root,registry:.builtIn(),idProvider:{serial+=1;return "cycle-\(serial)"})) {error in
            XCTAssertTrue(error.localizedDescription.contains("scope cycle"))
        }
        XCTAssertEqual(try WorkflowCodec.encode(original),bytes)
        let valid=try WorkflowInlineContainer.apply(.moveOut(childID:"branch-a",address:.init(domain:.model,localID:"outsideLinear")),to:original,root:root,registry:.builtIn(),idProvider:{serial+=1;return "cycle-\(serial)"})
        XCTAssertTrue(valid.nodes.contains {$0.address.localID=="outsideLinear"})
        XCTAssertEqual(valid.revision,original.revision+1)
        XCTAssertEqual(try WorkflowCodec.encode(original),bytes)
    }
    func testDisconnectedMemberIsEditableButCannotBeCompiledAsCompletedModel()throws {
        let original=try fixture()
        let authored=try WorkflowInlineContainer.apply(.insertNode(.init(id:"unconnected",type:"relu")),to:original,root:root,registry:.builtIn(),idProvider:{"incomplete-private"})
        XCTAssertTrue(authored.drafts.isEmpty)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(authored)),authored)
        XCTAssertNotNil(try WorkflowInlineContainer.node(in:authored,root:root,childID:"unconnected",registry:.builtIn()))
        let project=CompositeProjectDocument(schemaVersion:3,root:.init(nodes:authored.nodes.compactMap {node in
            guard case .modelPrimitive(let type,_,let config)=node.payload else {return nil}
            return GraphNode(id:node.address.localID,type:type,config:config)
        },instances:authored.nodes.compactMap {node in
            guard case .modelBlock(let reference,let overrides)=node.payload else {return nil}
            return CompositeInstance(id:node.address.localID,definition:reference,overrides:overrides)
        },edges:authored.edges.map {GraphEdge(id:$0.id,source:$0.from.address.localID,sourcePort:$0.from.port,target:$0.to.address.localID,targetPort:$0.to.port)}),definitions:authored.definitions)
        XCTAssertThrowsError(try CompositeExpander.expand(project,registry:.builtIn()),"Authoring acceptance must never imply execution completeness")
    }
}
