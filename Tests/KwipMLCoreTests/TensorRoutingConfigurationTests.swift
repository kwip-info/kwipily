import XCTest
@testable import KwipMLCore

final class TensorRoutingConfigurationTests:XCTestCase {
    private func layout(_ sample:[Int])throws->BatchedTensorLayout {try .init(sample:sample)}
    private func config(_ id:String,_ values:[String:JSONValue]=[:])throws->TensorRoutingConfiguration {try .init(componentID:id,values:values)}
    func testSplitSizeAndSectionsResolveEachActualOrderedPart()throws {
        let a=try layout([2,5]);let size=try config("split")
        let shapes=try size.resolved(layouts:[a])
        XCTAssertEqual(try size.outputPorts(layouts:[a]),["part0","part1","part2"])
        XCTAssertEqual(shapes["part0"]?.sample.dimensions,[2,2]);XCTAssertEqual(shapes["part2"]?.sample.dimensions,[2,1])
        let sections=try config("split",["mode":.string("sections"),"sections":.array([1,3,1])])
        XCTAssertEqual(try sections.resolved(layouts:[a])["part1"]?.sample.dimensions,[2,3])
        XCTAssertEqual(try config("split",["split_size":9]).outputPorts(layouts:[a]),["part0"])
        XCTAssertThrowsError(try config("split",["mode":.string("sections"),"sections":.array([1,2])]).resolved(layouts:[a]))
        XCTAssertThrowsError(try config("split",["split_size":1]).resolved(layouts:[layout([13])]))
    }
    func testChunkUsesTrueFewerThanRequestedNativeSections()throws {
        let c=try config("chunk",["chunks":6]);let s=try c.resolved(layouts:[layout([2,13])])
        XCTAssertEqual(try c.outputPorts(layouts:[layout([2,13])]),["part0","part1","part2","part3","part4"])
        XCTAssertEqual(s["part3"]?.sample.dimensions,[2,3]);XCTAssertEqual(s["part4"]?.sample.dimensions,[2,1]);XCTAssertNil(s["part5"])
        XCTAssertEqual(try config("chunk",["chunks":8]).outputPorts(layouts:[layout([2])]),["part0","part1"])
    }
    func testStrictSettingsDirectionsAndInactiveSemanticDefaults()throws {
        for id in ["split","chunk","gather"] {
            for bad in [0,true,1.5,8,-9] as [JSONValue] {XCTAssertThrowsError(try config(id,["dim":bad]))}
            XCTAssertThrowsError(try config(id,["dim":-3]).resolved(layouts:[layout([5])]))
            XCTAssertThrowsError(try config(id,["extra":1]))
        }
        for bad in [0,-1,true,1.5,100_000_001] as [JSONValue] {XCTAssertThrowsError(try config("split",["split_size":bad]))}
        for bad in [0,9,true,1.5] as [JSONValue] {XCTAssertThrowsError(try config("chunk",["chunks":bad]))}
        for bad in [JSONValue.array([]),.array([0,2]),.array([1,1.5]),.array([true,1])] {XCTAssertThrowsError(try config("split",["mode":.string("sections"),"sections":bad]))}
        XCTAssertEqual(try config("split",["split_size":2,"sections":.array([8,8])]).values,try config("split").values)
        XCTAssertEqual(try config("einsum",["pattern":.string("diagonal"),"equation":.string("na,nb->nab")]).values["equation"],.string("naa->na"))
    }
    func testGatherShapesAreTypedAndNeverBroadcastObservations()throws {
        let c=try config("gather",["dim":2]);let source=TypedTensorContract(layout:try layout([2,4]),dtype:.float32),indices=TypedTensorContract(layout:try layout([2,3]),dtype:.int64)
        XCTAssertEqual(c.configuredInputs,["source","indices"])
        XCTAssertEqual(try c.resolved(inputs:[source,indices])["out"],.init(layout:try layout([2,3]),dtype:.float32))
        XCTAssertThrowsError(try c.resolved(inputs:[source,source]))
        XCTAssertThrowsError(try c.resolved(layouts:[layout([2,4]),layout([3,3])]))
        XCTAssertThrowsError(try c.resolved(layouts:[layout([2,4]),layout([3])]))
    }
    func testEinsumPresetsIncludeRealUnaryAndBinarySemantics()throws {
        let m=try config("einsum");XCTAssertEqual(m.configuredInputs,["a","b"])
        XCTAssertEqual(try m.resolved(layouts:[layout([2,3]),layout([3,4])])["out"]?.sample.dimensions,[2,4])
        let diagonal=try config("einsum",["pattern":.string("diagonal")]);XCTAssertEqual(diagonal.configuredInputs,["a"])
        XCTAssertEqual(try diagonal.resolved(layouts:[layout([3,3])])["out"]?.sample.dimensions,[3])
        XCTAssertThrowsError(try diagonal.resolved(layouts:[layout([2,3])]))
        XCTAssertEqual(try config("einsum",["pattern":.string("outer")]).resolved(layouts:[layout([2]),layout([4])])["out"]?.sample.dimensions,[2,4])
        XCTAssertEqual(try config("einsum",["pattern":.string("reorder")]).resolved(layouts:[layout([2,3])])["out"]?.sample.dimensions,[3,2])
    }
    func testEinsumCustomEquationBroadcastContractionAndStrictGrammar()throws {
        let c=try config("einsum",["pattern":.string("custom"),"equation":.string(" n a b , n b c -> n a c ")])
        XCTAssertEqual(c.values["equation"],.string("nab,nbc->nac"))
        XCTAssertEqual(try c.resolved(layouts:[layout([2,1]),layout([3,4])])["out"]?.sample.dimensions,[2,4])
        let unary=try config("einsum",["pattern":.string("custom"),"equation":.string("nabc->na")])
        XCTAssertEqual(try unary.resolved(layouts:[layout([2,3,4])])["out"]?.sample.dimensions,[2])
        for equation in ["ab,bc->ac","nab,nbc","n...a->na","nab->n","nab->nqq","nab->naa","nab->anb","nab,nab,nab->nab","naa->na\n","nα->nα","->na"] {
            XCTAssertThrowsError(try config("einsum",["pattern":.string("custom"),"equation":.string(equation)]),equation)
        }
        XCTAssertThrowsError(try c.resolved(layouts:[layout([2,3]),layout([4,2])]))
        XCTAssertThrowsError(try c.resolved(layouts:[layout([1000,1000]),layout([1000,1000])]))
    }
    func testProductionTypedRoutingRefusesSelectorsAsTrainableFeatures()throws {
        let f=TypedTensorContract(layout:try layout([2,4]),dtype:.float32),i=TypedTensorContract(layout:try layout([2,4]),dtype:.int64)
        for id in ["split","chunk"] {XCTAssertThrowsError(try config(id).resolved(inputs:[i]));XCTAssertFalse(try config(id).resolved(inputs:[f]).isEmpty)}
        XCTAssertThrowsError(try config("einsum").resolved(inputs:[f,i]))
        XCTAssertThrowsError(try config("missing"))
    }
}
