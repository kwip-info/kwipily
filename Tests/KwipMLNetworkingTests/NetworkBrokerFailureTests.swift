import XCTest
@testable import KwipMLNetworking

final class NetworkBrokerFailureTests:XCTestCase {
    func testFailuresHaveSafeActionableDescriptionsWithoutInventingCause() {
        let cases:[(NetworkBrokerFailure,String)]=[
            (.unavailable,"The network service could not complete this request. Try again; if it continues, reopen Kwip ML."),
            (.signInRequired,"Authentication is required to complete this request."),
            (.deviceLimit,"This account has reached its device limit."),
            (.notEntitled,"This account does not have access to the requested service."),
            (.conflict,"The request conflicts with the current service state. Refresh and try again."),
            (.rateLimited,"The service is receiving too many requests. Wait before trying again.")]
        for (error,message) in cases {
            XCTAssertEqual(error.localizedDescription,message)
            XCTAssertFalse(error.localizedDescription.contains("error 0"))
        }
    }
}
