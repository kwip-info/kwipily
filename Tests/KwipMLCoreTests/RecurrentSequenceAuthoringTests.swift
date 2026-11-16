import XCTest
@testable import KwipMLCore

final class RecurrentSequenceAuthoringTests:XCTestCase {
    func testThreeSequenceMethodsPreserveSourceAndCellFamilyIsExcluded()throws {
        let registry=try ComponentRegistry.builtIn()
        XCTAssertEqual(RecurrentSequenceConfiguration.componentIDs,["rnn","lstm","gru"])
        for method in RecurrentSequenceConfiguration.Method.allCases {
            let config=try RecurrentSequenceConfiguration.defaults(for:method,registry:registry)
            XCTAssertEqual(config["input_size"],2);XCTAssertEqual(config["hidden_size"],2)
            let node=GraphNode(id:"memory",type:method.rawValue,name:"Sequence",x:19,y:37,config:[:])
            XCTAssertEqual(try RecurrentSequenceConfiguration.switching(node,configurationVersion:2,to:method,registry:registry),node)
        }
        let cell=GraphNode(id:"cell",type:"rnn_cell",config:[:])
        XCTAssertThrowsError(try RecurrentSequenceConfiguration.switching(cell,configurationVersion:2,to:.gru,registry:registry))
    }
    func testCannotRemoveConnectedCellStateOrSuppliedInputs()throws {
        let registry=try ComponentRegistry.builtIn()
        var config=try RecurrentSequenceConfiguration.defaults(for:.lstm,registry:registry);config["state_mode"] = .string("supplied")
        let node=GraphNode(id:"memory",type:"lstm",config:config)
        XCTAssertThrowsError(try RecurrentSequenceConfiguration.switching(node,configurationVersion:2,to:.gru,connectedOutputPorts:["cell"],registry:registry))
        XCTAssertThrowsError(try RecurrentSequenceConfiguration.switching(node,configurationVersion:2,to:.gru,connectedInputPorts:["cell"],registry:registry))
        config["state_mode"] = .string("zero")
        XCTAssertThrowsError(try RecurrentSequenceConfiguration.switching(node,configurationVersion:2,to:.lstm,configuration:config,connectedInputPorts:["hidden"],registry:registry))
        let changed=try RecurrentSequenceConfiguration.switching(node,configurationVersion:2,to:.gru,connectedInputPorts:["in","hidden"],connectedOutputPorts:["out","hidden"],registry:registry)
        XCTAssertEqual(changed.id,node.id);XCTAssertEqual(changed.type,"gru")
        XCTAssertEqual(changed.config["state_mode"],.string("supplied"))
    }
    func testPackedGatePoliciesAndRemovedNamedParametersRequireExplicitRepair()throws {
        let registry=try ComponentRegistry.builtIn()
        var node=GraphNode(id:"memory",type:"rnn",config:[:],parameterSettings:["weight_ih_l0":.object(["trainability":.string("frozen")])])
        XCTAssertThrowsError(try RecurrentSequenceConfiguration.switching(node,configurationVersion:2,to:.gru,registry:registry))
        node.type="lstm";node.config=["num_layers":2,"bidirectional":true]
        node.parameterSettings=["weight_ih_l1_reverse":.object(["trainability":.string("frozen")])]
        var config=try RecurrentSequenceConfiguration.defaults(for:.lstm,registry:registry)
        XCTAssertThrowsError(try RecurrentSequenceConfiguration.switching(node,configurationVersion:2,to:.lstm,configuration:config,registry:registry))
        config["num_layers"]=2;config["bidirectional"]=true
        let changed=try RecurrentSequenceConfiguration.switching(node,configurationVersion:2,to:.lstm,configuration:config,registry:registry)
        XCTAssertEqual(changed.parameterSettings,node.parameterSettings)
    }
    func testCanonicalGraphCommandRefusesInputRemovalAndKeepsExactBeforeState()throws {
        let registry=try ComponentRegistry.builtIn()
        let graph=GraphDocument(nodes:[.init(id:"state",type:"input",config:["features":2]),.init(id:"memory",type:"rnn",config:["state_mode":.string("supplied")])],edges:[.init(source:"state",target:"memory",targetPort:"hidden")])
        var config=try RecurrentSequenceConfiguration.defaults(for:.rnn,registry:registry)
        XCTAssertThrowsError(try GraphCommandEngine.apply(.setRecurrent(nodeID:"memory",method:.rnn,configuration:config),to:.init(document:graph),registry:registry))
        config["state_mode"] = .string("supplied")
        let result=try GraphCommandEngine.apply(.setRecurrent(nodeID:"memory",method:.rnn,configuration:config),to:.init(document:graph),registry:registry)
        XCTAssertEqual(result.before.document,graph);XCTAssertEqual(result.after.document.edges,graph.edges)
    }
}
