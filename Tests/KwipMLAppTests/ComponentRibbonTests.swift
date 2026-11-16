import XCTest
@testable import KwipMLApp

final class ComponentRibbonTests:XCTestCase {
    private var entries:[ComponentRibbonEntry] {[
        .init(id:"a",title:"Fixture source",shortTitle:"Source",purpose:"Adapter observations",category:"Source",typeIcon:"kwip.data.sourceCSV"),
        .init(id:"b",title:"Standard scaler",shortTitle:"Scale",purpose:"Fit training columns",category:"Fit",typeIcon:"kwip.data.preprocess"),
        .init(id:"c",title:"Apply fitted transform",shortTitle:"Apply",purpose:"Use fitted state",category:"Fit",typeIcon:"external.apply")
    ]}
    func testCategoriesOnlyContainAvailableEntriesInStableOrder() {
        XCTAssertEqual(ComponentRibbonCatalog.categories(entries),["Source","Fit"])
        XCTAssertEqual(ComponentRibbonCatalog.categories([]),[])
    }
    func testSearchCrossesSelectedCategoryAndMatchesPurposeWithoutCaseSensitivity() {
        XCTAssertEqual(ComponentRibbonCatalog.visible(entries,category:"Source",query:"  TRAINING  ").map(\.id),["b"])
        XCTAssertEqual(ComponentRibbonCatalog.visible(entries,category:"Fit",query:"").map(\.id),["b","c"])
        XCTAssertTrue(ComponentRibbonCatalog.visible(entries,category:nil,query:"unknown").isEmpty)
    }
    func testRemovedCategoryFallsBackToAvailableEntries() {
        XCTAssertEqual(ComponentRibbonCatalog.visible(entries,category:"Removed",query:"").map(\.id),["a","b","c"])
    }
    func testDataToolsFollowWorkflowOrderAcrossAllCategorySearchAndCommon() {
        let order=["kwip.data.sourceCSV","kwip.data.sourceFiles","kwip.data.columns","kwip.data.formula","kwip.data.filterRows","kwip.data.join","kwip.data.aggregate","kwip.data.sampleRows","kwip.data.partition","kwip.data.preprocess","kwip.data.view","kwip.data.outputDelimited","output"]
        let values=order.reversed().map {ComponentRibbonEntry(id:$0,title:$0,shortTitle:$0,purpose:"Data fixture",category:"Data",typeIcon:$0)}
        for scope:ComponentRibbonCatalog.Scope in [.all,.category("Data")] {
            XCTAssertEqual(ComponentRibbonCatalog.visible(values,scope:scope,query:"").map(\.id),order)
            XCTAssertEqual(ComponentRibbonCatalog.visible(values,scope:scope,query:"fixture").map(\.id),order)
        }
        let common=ComponentRibbonCatalog.visible(values,scope:.common,query:"").map(\.id)
        XCTAssertEqual(common,order.filter{common.contains($0)})
        let extensionEntry=ComponentRibbonEntry(id:"custom",title:"Custom",shortTitle:"Custom",purpose:"Extension",category:"Data",typeIcon:"external.tool")
        XCTAssertEqual(ComponentRibbonCatalog.visible(values+[extensionEntry],scope:.all,query:"").last?.id,"custom")
    }
    func testRetiredConsolidatedToolsCannotReappearThroughAllSearchOrCategory() {
        let retired=["kwip.data.missing","kwip.data.imputerFit","kwip.data.oneHotEncoderFit","kwip.data.standardScalerFit","kwip.data.transform","kwip.data.calculate","kwip.data.regexExtract","kwip.data.regexReplace","kwip.data.selectRename","kwip.data.cast","kwip.data.filter","kwip.data.filterExpression","kwip.data.regexMatch","dataset_source","data_loader","optimizer_sgd","scheduler_step_lr","loss_mse","initialization","gradient_policy","trainability"]
        let values=retired.map {ComponentRibbonEntry(id:$0,title:"Legacy \($0)",shortTitle:"Legacy",purpose:"Retired",category:"Old",typeIcon:$0)}
            + [ComponentRibbonEntry(id:"filter",title:"Filter",shortTitle:"Filter",purpose:"Current",category:"Data",typeIcon:"kwip.data.filterRows")]
        XCTAssertEqual(ComponentRibbonCatalog.visible(values,scope:.all,query:"").map(\.id),["filter"])
        XCTAssertTrue(ComponentRibbonCatalog.visible(values,scope:.all,query:"Legacy").isEmpty)
        XCTAssertEqual(ComponentRibbonCatalog.categories(values),["Data"])
        XCTAssertEqual(ComponentRibbonCatalog.visible(values,scope:.common,query:"").map(\.id),["filter"])
    }
}
