import XCTest
@testable import KwipMLCore

final class RecurrentCellAuthoringTests:XCTestCase {
    func testDefaultsExactSourceAndNoSequenceControls()throws {
        let registry=try ComponentRegistry.builtIn()
        for method in RecurrentCellConfiguration.Method.allCases {
            let type=method.rawValue+"_cell"
            let values=try RecurrentCellConfiguration.defaults(for:method,registry:registry)
            XCTAssertEqual(values["input_size"],2);XCTAssertEqual(values["hidden_size"],2)
            XCTAssertEqual(values["bias"],true);XCTAssertEqual(values["state_mode"],.string("zero"))
            for key in ["num_layers","dropout","bidirectional","proj_size"] {XCTAssertNil(values[key])}
            let original=GraphNode(id:"step",type:type,name:"Step",x:17,y:31,config:[:])
            XCTAssertEqual(try RecurrentCellConfiguration.switching(original,configurationVersion:2,to:method,registry:registry),original)
            var invalid=values;invalid["num_layers"]=2
            XCTAssertThrowsError(try RecurrentCellConfiguration(componentID:type,configurationVersion:2,values:invalid,registry:registry))
        }
        XCTAssertThrowsError(try RecurrentCellConfiguration(componentID:"rnn",configurationVersion:2,values:[:],registry:registry))
    }
    func testSwitchPreservesIdentitySharedControlsAndStatePorts()throws {
        let registry=try ComponentRegistry.builtIn()
        var config=try RecurrentCellConfiguration.defaults(for:.rnn,registry:registry)
        config["input_size"]=8;config["hidden_size"]=16;config["nonlinearity"] = .string("relu");config["state_mode"] = .string("supplied")
        let original=GraphNode(id:"step",type:"rnn_cell",name:"Step",x:17,y:31,config:config)
        let next=try RecurrentCellConfiguration.switching(original,configurationVersion:2,to:.lstm,connectedInputPorts:["in","hidden"],connectedOutputPorts:["out","hidden"],registry:registry)
        XCTAssertEqual(next.type,"lstm_cell");XCTAssertEqual(next.id,original.id);XCTAssertEqual(next.name,original.name);XCTAssertEqual(next.x,17)
        XCTAssertEqual(next.config["input_size"],8);XCTAssertEqual(next.config["hidden_size"],16);XCTAssertNil(next.config["nonlinearity"])
        XCTAssertThrowsError(try RecurrentCellConfiguration.switching(next,configurationVersion:2,to:.gru,connectedOutputPorts:["cell"],registry:registry))
        XCTAssertThrowsError(try RecurrentCellConfiguration.switching(next,configurationVersion:2,to:.gru,connectedInputPorts:["cell"],registry:registry))
        var zero=next.config;zero["state_mode"] = .string("zero")
        XCTAssertThrowsError(try RecurrentCellConfiguration.switching(next,configurationVersion:2,to:.lstm,configuration:zero,connectedInputPorts:["hidden"],registry:registry))
    }
    func testPoliciesAndParameterBudgetCannotBeSilentlyDiscarded()throws {
        let registry=try ComponentRegistry.builtIn()
        let original=GraphNode(id:"step",type:"rnn_cell",config:[:],parameterSettings:["bias_ih":.object(["trainability":.string("frozen")])])
        XCTAssertThrowsError(try RecurrentCellConfiguration.switching(original,configurationVersion:2,to:.gru,registry:registry))
        var values=try RecurrentCellConfiguration.defaults(for:.rnn,registry:registry);values["bias"]=false
        XCTAssertThrowsError(try RecurrentCellConfiguration.switching(original,configurationVersion:2,to:.rnn,configuration:values,registry:registry))
        XCTAssertThrowsError(try RecurrentCellConfiguration(componentID:"lstm_cell",configurationVersion:2,values:["input_size":4096,"hidden_size":512],registry:registry))
    }
    func testCanonicalCommandKeepsCellFamilyAndUndoBeforeState()throws {
        let registry=try ComponentRegistry.builtIn()
        let document=GraphDocument(nodes:[.init(id:"step",type:"rnn_cell",config:[:])],edges:[])
        let result=try GraphCommandEngine.apply(.setRecurrent(nodeID:"step",method:.gru),to:.init(document:document),registry:registry)
        XCTAssertEqual(result.before.document,document);XCTAssertEqual(result.after.document.nodes[0].type,"gru_cell")
        XCTAssertNil(result.after.document.nodes[0].config["num_layers"])
    }
    func testInlineCellBindingPreservesRawValuesAndExposedPorts()throws {
        let registry=try ComponentRegistry.builtIn(),root=WorkflowNodeAddress(domain:.model,localID:"group")
        var definition=CompositeDefinition(reference:.init("bound-cell"),title:"Step",body:.init(nodes:[.init(id:"memory",type:"rnn_cell",config:[:])],configurationVersion:2),inputs:[.init("in",targets:[.init("memory","in")])],outputs:[.init("out",source:.init("memory","out"))],parameters:[.init(.init(key:"width",title:"Width",defaultValue:8,min:1,max:512),targets:[.init("memory","hidden_size")])])
        definition.composition = .init(kind:.graph,children:[.init(localID:"memory")])
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:root,payload:.modelBlock(reference:definition.reference,overrides:[:]))],definitions:[definition])
        document.schemaVersion=11
        var config=try RecurrentCellConfiguration.defaults(for:.rnn,registry:registry);config["hidden_size"]=8;config["bias"]=false
        let edited=try WorkflowInlineContainer.apply(.setRecurrent(childID:"memory",method:.rnn,configuration:config),to:document,root:root,registry:registry,idProvider:{"private-cell"})
        let changed=try WorkflowInlineContainer.definition(in:edited,root:root)
        // The existing inline authoring migration fills omitted defaults before editing.
        // Keep that definition value (2), never the exposed effective override (8).
        XCTAssertEqual(changed.body.nodes[0].config["hidden_size"],2)
        XCTAssertEqual(changed.body.nodes[0].config["bias"],false)
        XCTAssertEqual(changed.body.nodes[0].type,"rnn_cell")
        config["hidden_size"]=9
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setRecurrent(childID:"memory",method:.rnn,configuration:config),to:document,root:root,registry:registry,idProvider:{"refused"}))
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setRecurrent(childID:"memory",method:.gru),to:document,root:root,registry:registry,idProvider:{"refused"}))
        XCTAssertEqual(document.definitions,[definition])
    }

}
