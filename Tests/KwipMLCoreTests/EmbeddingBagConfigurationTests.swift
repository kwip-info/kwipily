import XCTest
@testable import KwipMLCore

final class EmbeddingBagConfigurationTests:XCTestCase {
    func testDeclaredDefaultsAndConditionalRealInputs()throws {
        let original=EmbeddingBagConfiguration.defaults
        for packed in [false,true] {
            for weighted in [false,true] {
                var values=original
                values["layout"] = .string(packed ? "packed":"fixed")
                values["per_sample_weights"] = .bool(weighted)
                if weighted {values["mode"] = .string("sum")}
                let config=try EmbeddingBagConfiguration(values:values)
                XCTAssertEqual(config.inputPorts,["indices"]+(packed ? ["offsets"]:[])+(weighted ? ["per_sample_weights"]:[]))
                XCTAssertEqual(config.parameterCount,64)
                XCTAssertEqual(try EmbeddingBagConfiguration(values:config.values),config)
            }
        }
        XCTAssertEqual(original["layout"],.string("fixed"))
        XCTAssertEqual(original["mode"],.string("mean"))
    }
    func testNegativePaddingAndNumericControlsCanonicalizeWithoutLosingFlags()throws {
        var values=EmbeddingBagConfiguration.defaults
        values["num_embeddings"] = .number(5);values["embedding_dim"] = .number(3)
        values["padding_idx"] = -1;values["max_norm"] = .number(1.5)
        values["norm_type"] = 1;values["freeze"] = true;values["include_last_offset"] = true
        let result=try EmbeddingBagConfiguration(values:values)
        XCTAssertEqual(result.values["num_embeddings"],.integer(5))
        XCTAssertEqual(result.values["embedding_dim"],.integer(3))
        XCTAssertEqual(result.values["padding_idx"],4)
        XCTAssertEqual(result.values["norm_type"],.number(1))
        XCTAssertEqual(result.values["include_last_offset"],true)
        XCTAssertEqual(result.values["freeze"],true)
        XCTAssertEqual(result.parameterCount,15)
        values["padding_idx"] = -5
        XCTAssertEqual(try EmbeddingBagConfiguration(values:values).values["padding_idx"],0)
        XCTAssertEqual(values["padding_idx"],-5)
    }
    func testStrictFieldsTypesAndDimensions() {
        let original=EmbeddingBagConfiguration.defaults
        for key in original.keys {
            var values=original;values.removeValue(forKey:key)
            XCTAssertThrowsError(try EmbeddingBagConfiguration(values:values),key)
        }
        let invalid:[(String,JSONValue)]=[("extra",1),("num_embeddings",true),("num_embeddings",0),
            ("num_embeddings",.integer(Int64.max)),("embedding_dim",.number(1.5)),("embedding_dim",0),
            ("padding_idx",16),("padding_idx",-17),("padding_idx",true),
            ("mode",.string("median")),("mode",.null),("layout",.string("ragged")),("layout",false),
            ("freeze",1),("sparse",.null),("include_last_offset",1),
            ("per_sample_weights",1),("scale_grad_by_freq",.string("true"))]
        for (key,value) in invalid {
            var values=original;values[key]=value
            XCTAssertThrowsError(try EmbeddingBagConfiguration(values:values),key)
        }
        var values=original;values["num_embeddings"]=100_000_000;values["embedding_dim"]=1
        XCTAssertNoThrow(try EmbeddingBagConfiguration(values:values)) // Static algebra, no allocation grant.
        values["embedding_dim"]=2
        XCTAssertThrowsError(try EmbeddingBagConfiguration(values:values))
    }
    func testFloat32NormBoundariesRefuseUnderflowAndOverflow()throws {
        for key in ["norm_type","max_norm"] {
            for value in [JSONValue.bool(true),0,-1,.number(.nan),.number(.infinity),.number(1e-100),.number(Double(Float.greatestFiniteMagnitude)*2)] {
                var values=EmbeddingBagConfiguration.defaults;values[key]=value
                XCTAssertThrowsError(try EmbeddingBagConfiguration(values:values),key)
            }
            var values=EmbeddingBagConfiguration.defaults;values[key] = .number(Double(Float.leastNonzeroMagnitude))
            XCTAssertNoThrow(try EmbeddingBagConfiguration(values:values))
        }
    }
    func testInvalidNativeInteractionsRefuseEvenWhenFrozen()throws {
        for frozen in [false,true] {
            let patches:[[String:JSONValue]] = [
                ["mode":.string("max"),"sparse":true],["mode":.string("max"),"scale_grad_by_freq":true],
                ["mode":.string("mean"),"per_sample_weights":true],["mode":.string("max"),"per_sample_weights":true],
                ["mode":.string("sum"),"sparse":true,"scale_grad_by_freq":true]]
            for patch in patches {
                var values=EmbeddingBagConfiguration.defaults;values["freeze"] = .bool(frozen)
                values.merge(patch){_,new in new}
                XCTAssertThrowsError(try EmbeddingBagConfiguration(values:values),"\(patch)")
            }
        }
        for mode in ["sum","mean"] {
            var values=EmbeddingBagConfiguration.defaults;values["mode"] = .string(mode);values["sparse"]=true
            XCTAssertNoThrow(try EmbeddingBagConfiguration(values:values)) // No optimizer/device claim.
        }
    }
}
