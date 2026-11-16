import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataNodeSchemaProjectionTests:XCTestCase {
    private func node(_ id:String,_ op:String,_ config:[String:JSONValue]=[:])->DataFlowNode {.init(id:id,operatorID:"kwip.data."+op,config:config)}
    private func wire(_ source:String,_ target:String,_ output:String="table",_ input:String="table")->DataFlowEdge {.init(from:.init(node:source,port:output),to:.init(node:target,port:input))}
    private let names=[DataColumnSchema(name:" a,b ",dtype:"string",nullable:true),DataColumnSchema(name:"target",dtype:"string",nullable:true)]
    func testOCRDocumentsDeclarationMatchesGroupedScalarOutput()throws {
        var config=try OCRConfiguration.defaults();config["pageColumn"] = .string("custom_page")
        let graph=DataFlowDocument(nodes:[node("source","sourceFiles"),node("ocr","extractText",config),node("view","view")],edges:[wire("source","ocr"),wire("ocr","view","documents")])
        let input:[DataColumnSchema]=[.init(name:"file_id",dtype:"string"),.init(name:"custom_page",dtype:"int64"),.init(name:"prepared",dtype:"image"),.init(name:"prepared_source",dtype:"image"),.init(name:"prepared_to_source",dtype:"string"),.init(name:"label",dtype:"string")]
        let output=DataNodeSchemaProjection.upstreamSchema(for:"view",in:graph,sourceSchema:{_ in input})
        XCTAssertEqual(output.map(\.name),["file_id","label","document_text","ocr_status","ocr_page_count","ocr_region_count","ocr_review_count","ocr_pack"])
    }
    private func columns(_ rows:[(String,String,String)])->[String:JSONValue] {
        ["rows":.array(rows.map {source,name,type in .object(["source":.string(source),"included":.bool(true),"outputName":.string(name),"targetType":.string(type)])}),
         "castFailure":.string("error"),"trimText":.bool(false),"newColumns":.string("drop"),"missingPolicy":.string("preserve")]
    }
    private var filter:[String:JSONValue] {
        var config=FilterRowsConfigurationDraft.defaults
        config["conditions"] = .array([.object(["column":.string(" a,b "),"comparison":.string("isNotMissing"),"value":.null,"caseSensitive":.bool(true)])])
        return config
    }
    private var partition:[String:JSONValue] {["strategy":.string("random"),"seed":42,"validationPercent":20,"testPercent":20]}

    @MainActor func testSampleRowsPreservesTypedColumnsAndIsEditable() {
        let graph=DataFlowDocument(nodes:[node("source","sourceCSV"),node("sample","sampleRows",["count":20,"seed":42]),node("inspect","filterRows",filter)],edges:[wire("source","sample"),wire("sample","inspect")])
        XCTAssertEqual(DataNodeSchemaProjection.upstreamSchema(for:"inspect",in:graph,sourceSchema:{_ in self.names}),names)
        XCTAssertTrue(DataFlowWorkspaceStore.editableTypes.contains("kwip.data.sampleRows"))
    }
    func testExactNamesFollowConnectedSourceThroughSimultaneousRenameAndCast() {
        let graph=DataFlowDocument(nodes:[node("source","sourceCSV"),node("unrelated","sourceCSV"),
            node("columns","columns",columns([(" a,b ","target","float64"),("target"," a,b ","keep")])),
            node("inspect","filterRows",filter)],edges:[wire("source","columns"),wire("columns","inspect")])
        var reads:[String]=[]
        let actual=DataNodeSchemaProjection.upstreamSchema(for:"inspect",in:graph,sourceSchema:{id in reads.append(id);return id == "source" ? self.names : [.init(name:"WRONG",dtype:"bool")]})
        XCTAssertEqual(actual.map(\.name),["target"," a,b "])
        XCTAssertEqual(actual.map(\.dtype),["float64","string"])
        XCTAssertEqual(actual.map(\.nullable),[true,true])
        XCTAssertEqual(reads,["source"],"Selection elsewhere must not influence the connected schema.")
    }
    func testPartitionAndFilterPreserveSchemaWhilePreprocessUsesTrainingAncestry() {
        var rule=PreprocessConfiguration.Column(column:" a,b ");rule.scaling="standard"
        let graph=DataFlowDocument(nodes:[node("source","sourceCSV"),
            node("columns","columns",columns([(" a,b "," a,b ","int64"),("target","target","keep")])),
            node("filter","filterRows",filter),node("partition","partition",partition),
            node("prepare","preprocess",PreprocessConfiguration(columns:[rule]).values),node("inspect","filterRows",filter)],
            edges:[wire("source","columns"),wire("columns","filter"),wire("filter","partition","matching")]+["training","validation","test"].map {wire("partition","prepare",$0,$0)}+[wire("prepare","inspect","validation")])
        let input=DataNodeSchemaProjection.upstreamSchema(for:"prepare",in:graph,sourceSchema:{_ in self.names})
        XCTAssertEqual(input.map(\.dtype),["int64","string"])
        var reads:[String]=[]
        let actual=DataNodeSchemaProjection.upstreamSchema(for:"inspect",in:graph,sourceSchema:{id in reads.append(id);return self.names},capturedSchema:{endpoint in
            if endpoint == .init(node:"partition",port:"validation") || endpoint == .init(node:"partition",port:"test") {XCTFail("Projected learned schema must follow Training, not held-out declarations")}
            return nil
        })
        XCTAssertEqual(actual.map(\.name),[" a,b ","target"])
        XCTAssertEqual(actual.map(\.dtype),["float64","string"])
        XCTAssertEqual(actual.map(\.nullable),[true,true]);XCTAssertEqual(reads,["source"])
    }
    func testMeanImputationAndUnknownVocabularyRequireExplicitCapturedEvidence() {
        var mean=PreprocessConfiguration.Column(column:" a,b ");mean.missing="mean"
        var graph=DataFlowDocument(nodes:[node("source","sourceCSV"),
            node("columns","columns",columns([(" a,b "," a,b ","float64"),("target","target","keep")])),
            node("partition","partition",partition),node("prepare","preprocess",PreprocessConfiguration(columns:[mean]).values),node("inspect","filterRows",filter)],
            edges:[wire("source","columns"),wire("columns","partition")]+["training","validation","test"].map {wire("partition","prepare",$0,$0)}+[wire("prepare","inspect","validation")])
        let imputed=DataNodeSchemaProjection.upstreamSchema(for:"inspect",in:graph,sourceSchema:{_ in self.names})
        XCTAssertEqual(imputed.map(\.name),[" a,b ","target"])
        XCTAssertEqual(imputed.map(\.dtype),["float64","string"]);XCTAssertEqual(imputed.map(\.nullable),[false,true])
        graph.nodes[1]=node("columns","columns",columns([(" a,b "," a,b ","keep"),("target","target","keep")]))
        var category=PreprocessConfiguration.Column(column:" a,b ");category.encoding="oneHot"
        graph.nodes[3]=node("prepare","preprocess",PreprocessConfiguration(columns:[category]).values)
        XCTAssertTrue(DataNodeSchemaProjection.upstreamSchema(for:"inspect",in:graph,sourceSchema:{_ in self.names}).isEmpty)
        var sourceReads=0
        let captured=[DataColumnSchema(name:"observed category",dtype:"float64",nullable:false)]
        let actual=DataNodeSchemaProjection.upstreamSchema(for:"inspect",in:graph,sourceSchema:{_ in sourceReads+=1;return self.names},capturedSchema:{endpoint in endpoint == .init(node:"prepare",port:"validation") ? captured : nil})
        XCTAssertEqual(actual,captured);XCTAssertEqual(sourceReads,0)
    }
    func testCycleAndAmbiguousInputsRemainUnresolvedWithoutArbitrarySourceChoice() {
        let cycle=DataFlowDocument(nodes:[node("a","filterRows",filter),node("b","filterRows",filter)],edges:[wire("a","b","matching"),wire("b","a","matching")])
        XCTAssertTrue(DataNodeSchemaProjection.upstreamSchema(for:"a",in:cycle,sourceSchema:{_ in XCTFail("Cycle must not invent source evidence");return []}).isEmpty)
        let ambiguous=DataFlowDocument(nodes:[node("left","sourceCSV"),node("right","sourceCSV"),node("columns","columns",columns([(" a,b "," a,b ","keep")])),node("inspect","filterRows",filter)],edges:[wire("left","columns"),wire("right","columns"),wire("columns","inspect")])
        XCTAssertTrue(DataNodeSchemaProjection.upstreamSchema(for:"inspect",in:ambiguous,sourceSchema:{_ in XCTFail("Ambiguous ancestry must not choose a source");return self.names}).isEmpty)
    }
    func testExplicitEmptyCaptureDoesNotFallBackAndUnknownOperatorDoesNotInventSchema() {
        let graph=DataFlowDocument(nodes:[node("source","sourceCSV"),node("unknown","unknown"),node("inspect","filterRows",filter)],edges:[wire("source","unknown"),wire("unknown","inspect")])
        XCTAssertTrue(DataNodeSchemaProjection.upstreamSchema(for:"inspect",in:graph,sourceSchema:{_ in self.names}).isEmpty)
        XCTAssertTrue(DataNodeSchemaProjection.upstreamSchema(for:"source",in:graph,sourceSchema:{_ in XCTFail("Explicit empty capture is not absent evidence");return self.names},capturedSchema:{_ in []}).isEmpty)
    }
    func testRetiredCastDoesNotInferCurrentSchemaWithoutCapturedEvidence() {
        let graph=DataFlowDocument(nodes:[node("source","sourceCSV"),node("old","cast",["columns":.object([" a,b ":.string("float64")])]),node("inspect","filterRows",filter)],edges:[wire("source","old"),wire("old","inspect")])
        XCTAssertTrue(DataNodeSchemaProjection.upstreamSchema(for:"inspect",in:graph,sourceSchema:{_ in self.names}).isEmpty)
    }
}
