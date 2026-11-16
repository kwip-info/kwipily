import XCTest
@testable import KwipMLCore

final class LinearFamilyConfigurationTests:XCTestCase {
    func testModesPreserveOriginalDefaultsAndLegacySemantics()throws {
        let registry=try ComponentRegistry.builtIn()
        XCTAssertEqual(try LinearConfiguration.defaults(for:.explicit,registry:registry)["out_features"],8)
        XCTAssertEqual(try LinearConfiguration.defaults(for:.inferred,registry:registry)["out_features"],2)
        var legacy=GraphNode(id:"map",type:"linear",name:"Scores",x:17,y:29,config:["in_features":2,"out_features":6])
        legacy.config.removeValue(forKey:"bias")
        let effective=try LinearConfiguration(componentID:"linear",configurationVersion:1,values:legacy.config,registry:registry)
        XCTAssertEqual(effective.values["bias"],true)
        XCTAssertEqual(try LinearConfiguration.switching(legacy,configurationVersion:1,to:.explicit,registry:registry),legacy)
        var values=try LinearConfiguration.defaults(for:.inferred,registry:registry);values["out_features"]=6
        let changed=try LinearConfiguration.switching(legacy,configurationVersion:1,to:.inferred,configuration:values,connectedOutputPorts:["out"],registry:registry)
        XCTAssertEqual(changed.id,legacy.id);XCTAssertEqual(changed.name,legacy.name);XCTAssertEqual(changed.x,legacy.x)
        XCTAssertEqual(changed.type,"lazy_linear");XCTAssertNil(changed.config["in_features"])
        XCTAssertEqual(changed.config["out_features"],6);XCTAssertEqual(changed.config["bias"],true)
        XCTAssertEqual(legacy.type,"linear")
    }
    func testAuthoredPoliciesAndInvalidInputsCannotBeDiscarded()throws {
        let registry=try ComponentRegistry.builtIn()
        let settings:[String:JSONValue]=["weight":.object(["initialization":.object(["algorithm":.string("xavier_uniform"),"settings":.object(["gain":1])])])]
        let node=GraphNode(id:"map",type:"linear",config:try LinearConfiguration.defaults(for:.explicit,registry:registry),parameterSettings:settings)
        XCTAssertThrowsError(try LinearConfiguration.switching(node,configurationVersion:2,to:.inferred,registry:registry))
        var frozen=node;frozen.parameterSettings=["weight":.object(["trainability":.string("frozen")])]
        XCTAssertEqual(try LinearConfiguration.switching(frozen,configurationVersion:2,to:.inferred,registry:registry).parameterSettings,frozen.parameterSettings)
        var bias=node;bias.parameterSettings=["bias":.object(["trainability":.string("frozen")])]
        var config=node.config;config["bias"]=false
        XCTAssertThrowsError(try LinearConfiguration.switching(bias,configurationVersion:2,to:.explicit,configuration:config,registry:registry))
        config=node.config;config["in_features"]=0
        XCTAssertThrowsError(try LinearConfiguration.switching(node,configurationVersion:2,to:.explicit,configuration:config,registry:registry))
        config=node.config;config["out_features"]=513
        XCTAssertThrowsError(try LinearConfiguration.switching(node,configurationVersion:2,to:.explicit,configuration:config,registry:registry))
        XCTAssertThrowsError(try LinearConfiguration.switching(node,configurationVersion:2,to:.inferred,connectedOutputPorts:["indices"],registry:registry))
    }
    func testCanonicalCommandKeepsEdgesAndSupportsUndoSource()throws {
        let registry=try ComponentRegistry.builtIn()
        let graph=GraphDocument(nodes:[.init(id:"map",type:"lazy_linear",name:"Learned scores",config:["out_features":6,"bias":false]),.init(id:"out",type:"output")],edges:[.init(source:"map",target:"out")])
        let state=GraphEditorState(document:graph)
        var config=try LinearConfiguration.defaults(for:.explicit,registry:registry);config["out_features"]=6;config["bias"]=false
        let result=try GraphCommandEngine.apply(.setLinear(nodeID:"map",variant:.explicit,configuration:config),to:state,registry:registry)
        XCTAssertEqual(result.before,state);XCTAssertEqual(result.after.document.edges,graph.edges)
        XCTAssertEqual(result.after.document.nodes[0].type,"linear")
        XCTAssertEqual(result.after.document.nodes[0].name,"Learned scores")
    }
}
