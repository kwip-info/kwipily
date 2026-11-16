import XCTest
@testable import KwipMLApp

final class EpochChartAxisTests:XCTestCase {
    func testStartsAtFirstCapturedEpochAndIncludesEndpoints() {
        let axis=EpochChartAxis(epochs:[1,2,3])!
        XCTAssertEqual(axis.domain,1.0...3.0)
        XCTAssertEqual(axis.ticks,[1,2,3])
        let resumed=EpochChartAxis(epochs:[12,13,14])!
        XCTAssertEqual(resumed.domain,12.0...14.0)
        XCTAssertEqual(resumed.ticks.first,12)
        XCTAssertEqual(resumed.ticks.last,14)
        XCTAssertFalse(resumed.ticks.contains(0))
    }
    func testSingleObservationAndEmptyEvidenceDoNotInventEpochs() {
        XCTAssertNil(EpochChartAxis(epochs:[]))
        let axis=EpochChartAxis(epochs:[1])!
        XCTAssertEqual(axis.ticks,[1])
        XCTAssertTrue(axis.domain.contains(1))
        XCTAssertGreaterThan(axis.domain.lowerBound,0)
    }
    func testSparseAndLongCaptureTicksUseOnlyObservedEpochs() {
        let epochs=[1,10,30,50,100,200,400,800]
        let axis=EpochChartAxis(epochs:epochs)!
        XCTAssertEqual(axis.domain,1.0...800.0)
        XCTAssertLessThanOrEqual(axis.ticks.count,6)
        XCTAssertTrue(Set(axis.ticks).isSubset(of:Set(epochs)))
        XCTAssertEqual(axis.ticks.first,1)
        XCTAssertEqual(axis.ticks.last,800)
    }
}
