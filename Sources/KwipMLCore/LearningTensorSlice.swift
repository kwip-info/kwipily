import Foundation

/// A view of the final two axes; preceding coordinates remain explicit.
public struct LearningTensorMatrixSlice:Equatable {
    public let leadingCoordinates:[Int]
    public let matrix:LearningTensor
}
public extension LearningTensor {
    var matrixSliceCount:Int {
        guard wellFormed else {return 0}
        return shape.dropLast(2).reduce(1,*)
    }
    func matrixSlice(at index:Int)->LearningTensorMatrixSlice? {
        guard index>=0,index<matrixSliceCount else {return nil}
        let dimensions=shape.count == 1 ? [1,shape[0]] : Array(shape.suffix(2))
        let size=dimensions[0]*dimensions[1],start=index*size
        var remaining=index,coordinates=Array(repeating:0,count:max(0,shape.count-2))
        for axis in coordinates.indices.reversed() {
            coordinates[axis]=remaining%shape[axis];remaining/=shape[axis]
        }
        return .init(leadingCoordinates:coordinates,matrix:.init(shape:dimensions,values:Array(values[start..<(start+size)])))
    }
}
