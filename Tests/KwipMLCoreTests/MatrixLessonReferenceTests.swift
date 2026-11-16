import XCTest
@testable import KwipMLCore

final class MatrixLessonReferenceTests:XCTestCase {
    func testRowColumnProductsAndBothGradients()throws {
        let reference=try MatrixLessonReference(left:.init(shape:[1,2,2],values:[1,2,3,4]),right:.init(shape:[1,2,2],values:[5,6,7,8]))
        XCTAssertEqual(reference.output,.init(shape:[1,2,2],values:[19,22,43,50]))
        let gradients=try reference.gradients(for:.init(shape:[1,2,2],values:[1,2,3,4]))
        XCTAssertEqual(gradients.left.values,[17,23,39,53])
        XCTAssertEqual(gradients.right.values,[10,14,14,20])
    }
    func testBroadcastSumsGradientsAcrossHeadsButNotObservations()throws {
        let left=LearningTensor(shape:[2,2,1,2],values:[1,2,3,4,10,20,30,40])
        let right=LearningTensor(shape:[2,2,1],values:[5,6,7,8])
        let reference=try MatrixLessonReference(left:left,right:right)
        XCTAssertEqual(reference.output,.init(shape:[2,2,1,1],values:[17,39,230,530]))
        let gradients=try reference.gradients(for:.init(shape:[2,2,1,1],values:[1,2,0,0]))
        XCTAssertEqual(gradients.left.values,[5,6,10,12,0,0,0,0])
        XCTAssertEqual(gradients.right.values,[7,10,0,0])
    }
    func testRejectsInvalidShapesAndUnboundedLessonWork()throws {
        XCTAssertThrowsError(try MatrixLessonReference(left:.init(shape:[2,1,2],values:[1,2,3,4]),right:.init(shape:[1,2,1],values:[1,2])))
        XCTAssertThrowsError(try MatrixLessonReference(left:.init(shape:[1,2],values:[1,2]),right:.init(shape:[1,2,1],values:[1,2])))
        XCTAssertThrowsError(try MatrixLessonReference(left:.init(shape:[1,64,64],values:Array(repeating:1,count:4096)),right:.init(shape:[1,64,64],values:Array(repeating:1,count:4096))))
        let reference=try MatrixLessonReference(left:.init(shape:[1,1,1],values:[1]),right:.init(shape:[1,1,1],values:[2]))
        XCTAssertThrowsError(try reference.gradients(for:.init(shape:[1,1],values:[1])))
        XCTAssertThrowsError(try MatrixLessonReference(left:.init(shape:[1,1,1],values:[Double.greatestFiniteMagnitude]),right:.init(shape:[1,1,1],values:[2])))
    }
    func testBroadcastGradientAgainstFiniteDifferencesInBothRankDirections()throws {
        for shapes in [([2,2,1,2,3],[2,3,3,2]),([2,2,3],[2,3,3,2])] {
            let a=LearningTensor(shape:shapes.0,values:(0..<shapes.0.reduce(1,*)).map {Double($0%7-3)/4})
            let b=LearningTensor(shape:shapes.1,values:(0..<shapes.1.reduce(1,*)).map {Double($0%5-2)/3})
            let reference=try MatrixLessonReference(left:a,right:b)
            let g=LearningTensor(shape:reference.output.shape,values:reference.output.values.indices.map {Double($0%11-5)/6})
            let gradients=try reference.gradients(for:g)
            func loss(_ a:LearningTensor,_ b:LearningTensor)throws->Double {
                zip(try MatrixLessonReference(left:a,right:b).output.values,g.values).reduce(0) {$0+$1.0*$1.1}
            }
            for side in 0..<2 {
                let tensor=side == 0 ? a:b
                let gradient=side == 0 ? gradients.left:gradients.right
                for index in tensor.values.indices {
                    var plus=tensor,minus=tensor
                    plus.values[index]+=1e-5;minus.values[index]-=1e-5
                    let difference=try side == 0 ? loss(plus,b)-loss(minus,b):loss(a,plus)-loss(a,minus)
                    XCTAssertEqual(gradient.values[index],difference/2e-5,accuracy:1e-8)
                }
            }
        }
    }

}
