color_map<-function(blocks){
  grid=sqrt(blocks)
  probs<-seq(0,1,length.out = grid+1)
  a_quant<-quantile(0:256,probs)
  b_quant<-quantile(0:256,probs)
  mat<-NULL
  
  for (i in 1:grid){
    row<-rep(c((grid*(i-1)):((grid+grid*(i-1))-1)),each=256/grid)
    block<-matrix(rep(row,256/grid),ncol=256,byrow = TRUE)
    mat<-rbind(mat,block)
  }
  return(mat)
}



map<-color_map(16)

write.csv(map,'map.csv')
