import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowDataLoadingRunArgumentsTests:XCTestCase {
    func testSelectedFullRecipesUseExactWorkerFlags()throws {
        let dataset=DatasetRecipe(configuration:try .init(values:DatasetConfiguration.defaults),source:.init(domain:.control,localID:"rows"))
        let loader=DataLoaderRecipe(configuration:try .init(values:DataLoaderConfiguration.defaults),source:.init(domain:.control,localID:"groups"))
        let args=try WorkflowDataLoadingRunArguments.arguments(dataset:dataset,loader:loader)
        XCTAssertEqual(args.count,4)
        guard args.count==4 else {return}
        XCTAssertEqual(args[0],"--dataset-recipe-json");XCTAssertEqual(args[2],"--loader-recipe-json")
        XCTAssertEqual(try DatasetConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self,from:Data(args[1].utf8))),dataset.configuration)
        XCTAssertEqual(try DataLoaderConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self,from:Data(args[3].utf8))),loader.configuration)
    }
    func testAbsencePreservesArgumentsAndForeignSourceRefuses()throws {
        XCTAssertEqual(try WorkflowDataLoadingRunArguments.arguments(dataset:nil,loader:nil),[])
        let bad=DatasetRecipe(configuration:try .init(values:DatasetConfiguration.defaults),source:.init(domain:.data,localID:"rows"))
        XCTAssertThrowsError(try WorkflowDataLoadingRunArguments.arguments(dataset:bad,loader:nil))
    }
}
