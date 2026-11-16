import XCTest
@testable import KwipMLCore

final class LossConfigurationTests: XCTestCase {
    func testEightCanonicalDefaultsAndRecipeRoundTrip() throws {
        let expected:[String:[String:JSONValue]] = [
            "loss_cross_entropy":["reduction":.string("mean"),"target_mode":.string("class_indices"),"class_axis":-1,"class_weights":.array([]),"ignore_index":-100,"label_smoothing":.number(0)],
            "loss_bce_with_logits":["reduction":.string("mean"),"weight_mode":.string("none"),"weight_shape":.array([]),"weight_values":.array([]),"pos_weight_mode":.string("none"),"pos_weight_shape":.array([]),"pos_weight_values":.array([])],
            "loss_mse":["reduction":.string("mean"),"target_broadcast":false],"loss_l1":["reduction":.string("mean"),"target_broadcast":false],
            "loss_nll":["reduction":.string("mean"),"class_axis":-1,"class_weights":.array([]),"ignore_index":-100],
            "loss_kl_div":["reduction":.string("mean"),"log_target":false],"loss_ctc":["reduction":.string("mean"),"blank":0,"zero_infinity":false,"target_layout":.string("padded")],
            "loss_triplet_margin":["reduction":.string("mean"),"margin":.number(1),"p":.number(2),"eps":1e-6,"swap":false]
        ]
        XCTAssertEqual(LossConfiguration.componentIDs,Set(expected.keys))
        for (id,values) in expected {
            let c=try LossConfiguration(componentID:id,values:[:])
            XCTAssertEqual(c.values,values,id)
            XCTAssertEqual(try LossConfiguration.decodeRecipe(c.recipeValue),c)
            XCTAssertEqual(try LossConfiguration(componentID:id,values:c.values),c)
        }
    }
    func testAllMembersRetainMeaningfulNondefaultsAndReductions() throws {
        let choices:[String:[String:JSONValue]] = [
            "loss_cross_entropy":["target_mode":.string("probabilities"),"class_axis":1,"class_weights":.array([0,2]),"label_smoothing":1,"reduction":.string("none")],
            "loss_bce_with_logits":["weight_mode":.string("configured"),"weight_shape":.array([2]),"weight_values":.array([0,2]),"pos_weight_mode":.string("mapped"),"reduction":.string("sum")],
            "loss_mse":["target_broadcast":true,"reduction":.string("none")],"loss_l1":["target_broadcast":true,"reduction":.string("sum")],
            "loss_nll":["ignore_index":-2147483648,"class_weights":.array([1,0]),"class_axis":-7,"reduction":.string("sum")],
            "loss_kl_div":["log_target":true,"reduction":.string("batchmean")],"loss_ctc":["blank":511,"zero_infinity":true,"target_layout":.string("packed"),"reduction":.string("none")],
            "loss_triplet_margin":["margin":0.25,"p":0.5,"eps":0,"swap":true,"reduction":.string("none")]]
        for (id,settings) in choices {
            let c=try LossConfiguration(componentID:id,values:settings)
            for (key,value) in settings {
                if let number=value.doubleValue {XCTAssertEqual(c.values[key]?.doubleValue,number,"\(id).\(key)")}
                else if case .array(let expected)=value,case .array(let actual)=c.values[key],expected.allSatisfy({$0.doubleValue != nil}) {XCTAssertEqual(actual.map(\.doubleValue),expected.map(\.doubleValue),"\(id).\(key)")}
                else {XCTAssertEqual(c.values[key],value,"\(id).\(key)")}
            }
        }
        for id in LossConfiguration.componentIDs {
            for reduction in ["none","mean","sum"] {XCTAssertEqual(try LossConfiguration(componentID:id,values:["reduction":.string(reduction)]).values["reduction"],.string(reduction))}
            if id != "loss_kl_div" {XCTAssertThrowsError(try LossConfiguration(componentID:id,values:["reduction":.string("batchmean")]))}
        }
    }
    func testStrictTypesUnknownKeysAndNumericBounds() throws {
        for id in LossConfiguration.componentIDs {
            for config:[String:JSONValue] in [["unknown":1],["reduction":true],["reduction":.string("average")]] {XCTAssertThrowsError(try LossConfiguration(componentID:id,values:config),id)}
        }
        XCTAssertThrowsError(try LossConfiguration(componentID:"unknown",values:[:]))
        for v:JSONValue in [true,.number(0.5),-8,0,8] {XCTAssertThrowsError(try LossConfiguration(componentID:"loss_nll",values:["class_axis":v]))}
        for v:JSONValue in [true,.number(0.5),.integer(2147483648)] {XCTAssertThrowsError(try LossConfiguration(componentID:"loss_cross_entropy",values:["ignore_index":v]))}
        for v:JSONValue in [true,-0.1,1.01,.number(.nan),.null] {XCTAssertThrowsError(try LossConfiguration(componentID:"loss_cross_entropy",values:["label_smoothing":v]))}
        for id in ["loss_mse","loss_l1"] {XCTAssertThrowsError(try LossConfiguration(componentID:id,values:["target_broadcast":1]))}
        for v:JSONValue in [true,-1,512,.number(0.5)] {XCTAssertThrowsError(try LossConfiguration(componentID:"loss_ctc",values:["blank":v]))}
        XCTAssertThrowsError(try LossConfiguration(componentID:"loss_ctc",values:["zero_infinity":1]))
        XCTAssertThrowsError(try LossConfiguration(componentID:"loss_ctc",values:["target_layout":.string("ragged")]))
        XCTAssertThrowsError(try LossConfiguration(componentID:"loss_kl_div",values:["log_target":1]))
        for key in ["margin","p","eps"] {
            for v:JSONValue in [true,.null,.number(.infinity),.number(1e39),-0.1] {XCTAssertThrowsError(try LossConfiguration(componentID:"loss_triplet_margin",values:[key:v]))}
        }
        for key in ["margin","p"] {XCTAssertThrowsError(try LossConfiguration(componentID:"loss_triplet_margin",values:[key:0]))}
        XCTAssertThrowsError(try LossConfiguration(componentID:"loss_triplet_margin",values:["p":16.01]))
        XCTAssertEqual(try LossConfiguration(componentID:"loss_triplet_margin",values:["margin":.number(LossConfiguration.scalarMinimumPositive),"p":16]).values["p"]?.doubleValue,16)
    }
    func testWeightGeometryAndDormantControlsRemainStrict() throws {
        for v:JSONValue in [.array([-1]),.array([true]),.array([.number(.infinity)]),.array(Array(repeating:1,count:513))] {XCTAssertThrowsError(try LossConfiguration(componentID:"loss_nll",values:["class_weights":v]))}
        let dormant=try LossConfiguration(componentID:"loss_bce_with_logits",values:["weight_mode":.string("none"),"weight_shape":.array([2]),"weight_values":.array([1,3])])
        XCTAssertEqual(dormant.values["weight_values"],.array([.number(1),.number(3)]))
        let scalar=try LossConfiguration(componentID:"loss_bce_with_logits",values:["weight_mode":.string("configured"),"weight_values":.array([2])])
        XCTAssertEqual(scalar.values["weight_shape"],.array([]))
        for bad:[String:JSONValue] in [["weight_mode":.string("configured")],["weight_shape":.array([2]),"weight_values":.array([1]),"weight_mode":.string("configured")],["weight_shape":.array([0])],["pos_weight_values":.array([-1])],["weight_shape":.array([true])],["weight_values":.array([true])],["weight_mode":.string("automatic")]] {XCTAssertThrowsError(try LossConfiguration(componentID:"loss_bce_with_logits",values:bad))}
        XCTAssertThrowsError(try LossConfiguration(componentID:"loss_cross_entropy",values:["target_mode":.string("probabilities"),"ignore_index":0]))
    }
    func testExactEnvelopeVersionsAndRecipeSize() throws {
        let config=try LossConfiguration(componentID:"loss_mse",values:[:])
        guard case .object(let original)=config.recipeValue else {return XCTFail("Object recipe")}
        for (key,value) in [("extra",JSONValue.integer(1)),("schemaVersion",2),("configurationVersion",2),("schemaVersion",true)] {
            var changed=original;changed[key]=value
            XCTAssertThrowsError(try LossConfiguration.decodeRecipe(.object(changed)))
        }
        var missing=original;missing.removeValue(forKey:"configurationVersion")
        XCTAssertThrowsError(try LossConfiguration.decodeRecipe(.object(missing)))
        let large=JSONValue.array(Array(repeating:.number(1.234567890123456e20),count:4096))
        XCTAssertThrowsError(try LossConfiguration(componentID:"loss_bce_with_logits",values:["weight_shape":.array([4096]),"weight_values":large,"weight_mode":.string("configured")]))
    }
    func testIntegralRealWeightWireBudgetMatchesRuntimeCanonicalSpelling() throws {
        var values=try LossConfiguration(componentID:"loss_bce_with_logits",values:[:]).values
        values["weight_mode"] = .string("configured")
        values["weight_shape"] = .array([4096])
        values["weight_values"] = .array(Array(repeating:.number(100000),count:4096))
        let wire:JSONValue = .object(["schemaVersion":1,"componentID":.string("loss_bce_with_logits"),"configurationVersion":1,"config":.object(values)])
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        XCTAssertLessThan(try encoder.encode(wire).count,LossConfiguration.maximumRecipeBytes,"Fixture exercises compact Swift integral-real spelling, not an already oversized wire.")
        XCTAssertThrowsError(try LossConfiguration(componentID:"loss_bce_with_logits",values:values))
        XCTAssertThrowsError(try LossConfiguration.decodeRecipe(wire))
        values["weight_values"] = .array(Array(repeating:.number(0),count:4096))
        XCTAssertNoThrow(try LossConfiguration(componentID:"loss_bce_with_logits",values:values),"A bounded integral-valued weight list remains supported.")
    }
    func testTargetBindingRejectsWrongObjectiveTypesAndUnselectedRoles() throws {
        func mapping(_ id:String,_ dtype:ContractDType,_ shape:[Int]=[])->LossTargetMapping {.init(componentID:id,inputColumns:["x"],roles:[.init(role:"target",columns:shape == [2] ? ["y0","y1"]:["y"],sampleShape:shape,dtype:dtype)])}
        let ce=try LossConfiguration(componentID:"loss_cross_entropy",values:[:])
        XCTAssertNoThrow(try ce.validate(targetMapping:mapping("loss_cross_entropy",.int64)))
        XCTAssertThrowsError(try ce.validate(targetMapping:mapping("loss_mse",.float32)))
        XCTAssertThrowsError(try ce.validate(targetMapping:mapping("loss_cross_entropy",.float32,[2])))
        let shares=try LossConfiguration(componentID:"loss_cross_entropy",values:["target_mode":.string("probabilities")])
        XCTAssertNoThrow(try shares.validate(targetMapping:mapping("loss_cross_entropy",.float32,[2])))
        XCTAssertThrowsError(try shares.validate(targetMapping:mapping("loss_cross_entropy",.int64)))
        var bce=mapping("loss_bce_with_logits",.float32)
        bce.roles.append(.init(role:"weight",columns:["importance"],sampleShape:[],dtype:.float32))
        XCTAssertThrowsError(try LossConfiguration(componentID:"loss_bce_with_logits",values:[:]).validate(targetMapping:bce))
        XCTAssertNoThrow(try LossConfiguration(componentID:"loss_bce_with_logits",values:["weight_mode":.string("mapped")]).validate(targetMapping:bce))
        XCTAssertThrowsError(try LossConfiguration(componentID:"loss_bce_with_logits",values:["pos_weight_mode":.string("mapped")]).validate(targetMapping:bce))
    }
    func testQualifiedRecipeValidatesActualMappingWithoutFakeTarget() throws {
        let c=try LossConfiguration(componentID:"loss_triplet_margin",values:[:])
        let mapping=LossTargetMapping(componentID:"loss_triplet_margin",inputColumns:["a","p","n"],roles:[],tripletSourceGroups:[["a"],["p"],["n"]])
        let source=WorkflowNodeAddress(domain:.control,localID:"objective")
        let recipe=try LossRecipe(configuration:c,source:source,targetMapping:mapping)
        XCTAssertEqual(recipe.source,source);XCTAssertEqual(recipe.targetMapping,mapping)
        var broken=mapping;broken.tripletSourceGroups=[["p"],["a"],["n"]]
        XCTAssertThrowsError(try LossRecipe(configuration:c,source:source,targetMapping:broken))
    }
}
