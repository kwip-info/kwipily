import XCTest
import CryptoKit
@testable import KwipMLCore

final class WebhookContractTests:XCTestCase {
    func testSignedRequestRejectsTamperingAndStaleTimestamp()throws {
        let body=Data("{\"items\":[]}".utf8),secret=String(repeating:"s",count:43),timestamp="1792425600",id="order-42"
        let signature=WebhookAuthentication.signature(secret:secret,timestamp:timestamp,deliveryID:id,body:body)
        let now=Date(timeIntervalSince1970:1792425600)
        try WebhookAuthentication.verify(method:.signed,secret:secret,headers:["x-kwip-timestamp":timestamp,"x-kwip-delivery-id":id,"x-kwip-signature":signature],body:body,now:now)
        XCTAssertThrowsError(try WebhookAuthentication.verify(method:.signed,secret:secret,headers:["x-kwip-timestamp":timestamp,"x-kwip-delivery-id":id,"x-kwip-signature":signature],body:Data("{}".utf8),now:now))
        XCTAssertThrowsError(try WebhookAuthentication.verify(method:.signed,secret:secret,headers:["x-kwip-timestamp":timestamp,"x-kwip-delivery-id":id,"x-kwip-signature":signature],body:body,now:now.addingTimeInterval(301)))
    }
    func testMappingRequiresTypedFieldsAndProducesOnlySelectedColumns()throws {
        let mapping=WebhookPayload(tableNodeID:"input",recordsPath:"items",columns:[.init(field:"id",column:"order_id",type:.int64),.init(field:"amount",column:"amount",type:.float64)])
        let output=try mapping.decode(Data("{\"items\":[{\"id\":42,\"amount\":2.5,\"secret\":\"omit\"}]}".utf8))
        XCTAssertEqual(output.rows,1)
        XCTAssertEqual(try JSONDecoder().decode(JSONValue.self,from:output.data),.array([.object(["order_id":42,"amount":2.5])]))
        XCTAssertThrowsError(try mapping.decode(Data("{\"items\":[{\"id\":\"42\",\"amount\":2.5}]}".utf8)))
        XCTAssertThrowsError(try mapping.decode(Data("{\"items\":[{\"id\":42}]}".utf8)))
    }
    func testBearerAndDeliveryIdentityValidation()throws {
        let token=String(repeating:"a",count:43)
        try WebhookAuthentication.verify(method:.bearer,secret:token,headers:["authorization":"Bearer "+token,"x-kwip-delivery-id":"stable-1"],body:Data())
        XCTAssertThrowsError(try WebhookAuthentication.verify(method:.bearer,secret:token,headers:["authorization":"Bearer "+token],body:Data()))
        XCTAssertThrowsError(try WebhookAuthentication.verify(method:.bearer,secret:token,headers:["authorization":"Bearer wrong","x-kwip-delivery-id":"1"],body:Data()))
    }
}
